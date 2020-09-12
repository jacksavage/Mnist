using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace mnist
{
    static class Utilities
    {
        public static readonly Random Rand = new Random();

        public static T[] ArrayFill<T>(int size, Func<T> func)
        {
            var items = new T[size];
            for (int i = 0; i < items.Length; i++)
                items[i] = func();
            return items;
        }

        public static IEnumerable<T> SelectRange<T>(this T[] source, int startIndex, int count) =>
            Enumerable.Range(startIndex, count)
            .Where(i => i < source.Length)
            .Select(i => source[i]);

        // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
        public static void Shuffle<T>(this T[] items)
        {
            for (var i = items.Length; i > 1; i--)
                items.Swap(i - 1, Rand.Next(i));
        }

        private static void Swap<T>(this T[] items, int i, int j) =>
            (items[i], items[j]) = (items[j], items[i]);

        public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        public static TC[] ArrayZip<TA, TB, TC>(TA[] a, TB[] b, Func<TA, TB, TC> func)
        {
            if (a.Length != b.Length) return null;
            var result = new TC[a.Length];

            for (var i = 0; i < a.Length; i++)
                result[i] = func(a[i], b[i]);

            return result;
        }

        public static TB[] ArraySelect<TA, TB>(TA[] a, Func<TA, TB> func)
        {
            var result = new TB[a.Length];
            for (var i = 0; i < a.Length; i++)
                result[i] = func(a[i]);
            return result;
        }
    }
}
