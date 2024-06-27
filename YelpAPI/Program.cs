using System;
using System.Linq;
using System.Net;
using System.Reactive.Subjects;
using System.Reactive.Concurrency;
using System.Reactive.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using YelpAPI;
using Yelp.Api;
using Yelp.Api.Models;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.IO;

public class SentimentAnalysisService
{
    private readonly MLContext _mlContext;
    private readonly PredictionEngine<SentimentData, SentimentPrediction> _predictionEngine;
    private readonly Subject<SentimentAnalysisResult> _sentimentSubject;

    public IObservable<SentimentAnalysisResult> SentimentStream => _sentimentSubject;

    public SentimentAnalysisService()
    {
        _mlContext = new MLContext();
        var model = TrainModel();
        _predictionEngine = _mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        _sentimentSubject = new Subject<SentimentAnalysisResult>();
    }

    private ITransformer TrainModel()
    {
        // load csv fajla sa podacima za treniranje

        string dir = Directory.GetCurrentDirectory();
        var data = _mlContext.Data.LoadFromTextFile<SentimentData>(Path.Combine(dir, "DataSet.csv"), separatorChar: ',', hasHeader: true);

        // data processing pipeline koji:
        // konvertuje 'SentimentText' u numerical features koriscenjem 'FeaturizeText'
        // trenira logisticki regresioni model koriscenjem 'SdcaLogisticRegression' gde je label column 'Sentiment' a feature column 'Features'
        // label column je onaj koji se prediktuje, dok je feature column onaj na osnovu kojeg se prediktuje

        var pipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
            .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(SentimentData.Sentiment), featureColumnName: "Features"));

        // vraca istrenirani model ('ITransformer')

        return pipeline.Fit(data);
    }

    public object AnalyzeSentiment(IList<string> comments)
    {
        var sentimentDataList = new List<SentimentAnalysisResult>();

        foreach (var comment in comments)
        {
            // predikcija svakog komentara i dodavanje sentimentResult-a u sentimentDataList
            var prediction = _predictionEngine.Predict(new SentimentData { SentimentText = comment });

            var sentimentResult = new SentimentAnalysisResult
            {
                SentimentText = comment,
                Sentiment = prediction.Prediction,
                Score = prediction.Score
            };

            sentimentDataList.Add(sentimentResult);

            // emit preko _sentimentSubject
            _sentimentSubject.OnNext(sentimentResult);
        }

        // kalkulacije razlicitih score-ova
        var totalScore = sentimentDataList.Sum(data => data.Score);
        var averageScore = totalScore / sentimentDataList.Count;

        bool averageSentiment = averageScore > 0;

        // sortiranje po Score
        var sortedSentiments = sentimentDataList.OrderByDescending(data => data.Score);

        // najpozitivniji, najnegativniji komentar
        var mostPositiveComment = sortedSentiments.First();
        var mostNegativeComment = sortedSentiments.Last();

        // summary sa svim rezultatima, prosekom, najpozitivnijim i najnegativnijim komentarom
        var result = new
        {
            Sentiments = sentimentDataList,
            Summary = new
            {
                AverageSentiment = averageScore,
                MostPositiveComment = new
                {
                    mostPositiveComment.SentimentText,
                    mostPositiveComment.Sentiment,
                    mostPositiveComment.Score
                },
                MostNegativeComment = new
                {
                    mostNegativeComment.SentimentText,
                    mostNegativeComment.Sentiment,
                    mostNegativeComment.Score
                }
            }
        };

        return result;
    }
}


// data klase

// struktura za treniranje modela
public class SentimentData
{
    [LoadColumn(0)]
    public float ItemID { get; set; }

    [LoadColumn(1)]
    public bool Sentiment { get; set; }

    [LoadColumn(2)]
    public string? SentimentText { get; set; }

    [LoadColumn(3)]
    public float Score { get; set; }
}

// struktura prediction output-a modela
public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    public float Score { get; set; }
}

// struktura rezultata analize za jedan komentar
public class SentimentAnalysisResult
{
    public string? SentimentText { get; set; }
    public bool Sentiment { get; set; }
    public float Score { get; set; }
}

namespace YelpAPI
{

    public class BusinessStream : IObservable<Business>
    {
        private readonly Subject<Business> businessSubject;

        public BusinessStream()
        {
            businessSubject = new Subject<Business>();
        }

        public void GetBusinesses(string location, string apiKey, IScheduler scheduler, MLContext mlContext)
        {
            var client = new YelpClient(apiKey);
            Observable.Start(async () =>
            {
                try
                {
                    var request = new SearchRequest
                    {
                        OpenNow = true,
                        Location = location,
                        Categories = "food"
                    };

                    var searchResults = await client.SearchBusinessesAllAsync(request);
                    var businesses = searchResults.Businesses.ToList();

                    foreach (var business in businesses)
                    {
                        List<string> comments = await GetBusinessComments(client, business.Id);
                        float averageSentiment = await CalculateAverageSentiment(mlContext, comments);

                        var newBusiness = new Business
                        {
                            Name = business.Name,
                            Price = business.Price,
                            Rating = business.Rating,
                            AverageSentiment = averageSentiment
                        };

                        businessSubject.OnNext(newBusiness);
                    }

                    businessSubject.OnCompleted();
                }
                catch (Exception ex)
                {
                    businessSubject.OnError(ex);
                }

            }, scheduler);
        }

        public IDisposable Subscribe(IObserver<Business> observer)
        {
            return businessSubject.Subscribe(observer);
        }

        private async Task<List<string>> GetBusinessComments(YelpClient client, string businessId)
        {
            var reviews = await client.GetReviewsAsync(businessId);
            return reviews.Reviews.Select(r => r.Text).ToList();
        }

        private async Task<float> CalculateAverageSentiment(MLContext mlContext, List<string> comments)
        {
            var sentimentService = new SentimentAnalysisService(mlContext);

            var sentimentResults = comments.Select(comment => sentimentService.AnalyzeSentiment(comment)).ToList();

            float totalScore = sentimentResults.Sum(result => result.Score);
            float averageScore = totalScore / sentimentResults.Count;

            return averageScore;
        }
    }


    public class Business
    {
        public string Name { get; set; }
        public string Price { get; set; }
        public float Rating { get; set; }
        public float AverageSentiment { get; set; }
    }

    public class BusinessObserver : IObserver<Business>
    {
        private readonly string name;

        public BusinessObserver(string name)
        {
            this.name = name;
        }

        public void OnNext(Business business)
        {
            Console.WriteLine($"{name}: {business.Name} | Rating: {business.Rating} | Price: {business.Price} | Average Sentiment: {business.AverageSentiment}");
        }

        public void OnError(Exception e)
        {
            Console.WriteLine($"{name}: Error happened: {e.Message}");
        }

        public void OnCompleted()
        {
            Console.WriteLine($"{name}: All businesses returned successfully!");
        }
    }

    public class HttpServer
    {
        private readonly string url;
        private BusinessStream? businessStream;
        private IDisposable? subscription1;
        private IDisposable? subscription2;
        private IDisposable? subscription3;

        public HttpServer(string url)
        {
            this.url = url;
        }

        public void Start()
        {
            var listener = new HttpListener();
            listener.Prefixes.Add(url);

            listener.Start();
            Console.WriteLine("Server started. Listening for incoming requests...");

            while (true)
            {
                var context = listener.GetContext();
                Task.Run(() => HandleRequest(context));
            }
        }

        private void HandleRequest(HttpListenerContext context)
        {
            var request = context.Request;
            var response = context.Response;
            byte[] buffer;

            if (request.HttpMethod == "GET")
            {
                string location = request.QueryString["location"];
                string apiKey = "NVp8e7MgMkyTPZ8GzG8-80b_7nln_hwIxInTc8TIjXBYSbo6bUEuH7q4iR078rTZsJoNvDiHdvSPo5vnBKOrvU3obQGLOx7ZOSW3Q5hNAgKE8l2DLJX2WX3xLm51ZnYx";

                if (String.IsNullOrEmpty(location))
                {
                    response.StatusCode = (int)HttpStatusCode.BadRequest;
                    buffer = Encoding.UTF8.GetBytes("Bad request! 'location' parameter is required.");
                    response.ContentLength64 = buffer.Length;
                    response.OutputStream.Write(buffer, 0, buffer.Length);
                    response.OutputStream.Close();
                }
                else
                {
                    IScheduler scheduler = NewThreadScheduler.Default;

                    using (var mlContext = new MLContext())
                    {
                        businessStream = new BusinessStream();
                        var observer1 = new BusinessObserver("Observer 1");
                        var observer2 = new BusinessObserver("Observer 2");
                        var observer3 = new BusinessObserver("Observer 3");

                        var filteredStream = businessStream;

                        subscription1 = filteredStream.Subscribe(observer1);
                        subscription2 = filteredStream.Subscribe(observer2);
                        subscription3 = filteredStream.Subscribe(observer3);

                        businessStream.GetBusinesses(location, apiKey, scheduler, mlContext);

                        response.StatusCode = (int)HttpStatusCode.OK;
                        buffer = Encoding.UTF8.GetBytes("Request received. Processing businesses...");
                        response.ContentLength64 = buffer.Length;
                        response.OutputStream.Write(buffer, 0, buffer.Length);
                        response.OutputStream.Close();
                    }
                }
            }
            else
            {
                response.StatusCode = (int)HttpStatusCode.NotFound;
                response.OutputStream.Close();
            }
        }

        public void Stop()
        {
            subscription1?.Dispose();
            subscription2?.Dispose();
            subscription3?.Dispose();
        }
    }

    internal class Program
    {
        public static void Main()
        {
            HttpServer server;
            string url = "http://localhost:8080/";
            server = new HttpServer(url);
            server.Start();
        }
    }
}