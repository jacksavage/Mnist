using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using Serilog;

namespace mnist
{
    public record Pattern(double[] X, double[] Y)
    {
        public static Pattern[] LoadPatterns(string baseUrl, string imageFile, string labelFile)
        {
            return Enumerable.Zip(
                DownloadParseData(baseUrl + imageFile, ReadImages),
                DownloadParseData(baseUrl + labelFile, ReadLabels),
                (x, y) => new Pattern(x, y)
            ).ToArray();
        }

        private static IEnumerable<double[]> DownloadParseData(string url, Func<BinaryReader, IEnumerable<double[]>> parser)
        {
            // download the file if it doesn't exist
            var name = Path.GetFileName(url);
            if (!File.Exists(name))
            {
                Log.Information("Downloading {Url}", name, url);
                using (var client = new WebClient())
                    client.DownloadFile(url, name);

                if (!File.Exists(name))
                {
                    Log.Warning("Failed to download {FileName}", name);
                    yield break;
                }
            }

            // open the file as a binary reader
            Log.Information("Reading from {FileName}", name);
            using var file = File.OpenRead(name);
            using var gzip = new GZipStream(file, CompressionMode.Decompress);
            using var reader = new BinaryReader(gzip);
            foreach (var vec in parser(reader)) yield return vec;
        }

        private static IEnumerable<double[]> ReadImages(BinaryReader reader)
        {
            // read the header
            if (ReadInt(reader) != 2051) yield break;  // magic number
            var n = ReadInt(reader);                   // sample count
            var h = ReadInt(reader);                   // image height
            var w = ReadInt(reader);                   // image width
            var m = h * w;                              // pixels per image

            // read each image
            for (int i = 0; i < n; i++)
            {
                yield return reader
                    .ReadBytes(m)                               // read the bytes for each pixel
                    .Select(p => p / (double)byte.MaxValue)     // normalize between 0 and 1
                    .ToArray();
            }
        }

        private static IEnumerable<double[]> ReadLabels(BinaryReader reader)
        {
            // read the header
            if (ReadInt(reader) != 2049) yield break;     // magic number
            var n = ReadInt(reader);                      // sample count

            // read each label
            for (int i = 0; i < n; i++)
            {
                // convert digit label to a one-hot vector
                var b = reader.ReadByte();
                yield return Enumerable.Range(0, 10)
                    .Select(i => b == i ? 1.0 : 0.0)
                    .ToArray();
            }
        }

        private static int ReadInt(BinaryReader reader)
        {
            if (!BitConverter.IsLittleEndian) return reader.ReadInt32();
            var bytes = reader.ReadBytes(4).Reverse().ToArray();
            return BitConverter.ToInt32(bytes);
        }
    }
}
