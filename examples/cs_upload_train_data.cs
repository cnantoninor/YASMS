class Program
{
    static Main(string[] args)
    {
        var url = "http://85.235.146.174:8000/models/spam_classifier/GradientBoostingClassifier/test_nino/upload_train_data";
        using (var client = new HttpClient())
        {
            using (var multipartFormDataContent = new MultipartFormDataContent())
            {
                var bytes = await System.IO.File.ReadAllBytesAsync("model_data.csv");
                var byteArrayContent = new ByteArrayContent(bytes);
                byteArrayContent.Headers.Add("Content-Type", "text/csv");
                multipartFormDataContent.Add(byteArrayContent, "train_data", "model_data.csv");

                multipartFormDataContent.Add(new StringContent("Testo"), "features_fields");
                multipartFormDataContent.Add(new StringContent("Stato Workflow"), "target_field");

                client.DefaultRequestHeaders.Add("accept", "application/json");

                var response = await client.PostAsync(url, multipartFormDataContent);
                var responseContent = await response.Content.ReadAsStringAsync();

                Console.WriteLine(responseContent);
            }
        }
    }
}