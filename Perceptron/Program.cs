using System;

namespace perceptron
{
    public class Program
    {
        static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        static double SigmoidDeriv(double y) => y * (1.0 - y); 

        public static void Main(string[] args)
        {
            double[,] datos = { { 1, 1, 0 }, { 1, 0, 1 }, { 0, 1, 1 }, { 0, 0, 0 } };

            Random rango = new(42);
            double lr = 0.5;
            double[,] w1 = {
                { rango.NextDouble() - 0.5, rango.NextDouble() - 0.5 },
                { rango.NextDouble() - 0.5, rango.NextDouble() - 0.5 }
            };
            double[] b1 = [ rango.NextDouble() - 0.5, rango.NextDouble() - 0.5 ];
            double[] w2 = [ rango.NextDouble() - 0.5, rango.NextDouble() - 0.5 ];
            double b2 = rango.NextDouble() - 0.5;

            for (int epoca = 0; epoca < 10000; epoca++)
            {
                for (int i = 0; i < 4; i++)
                {
                    double x0 = datos[i, 0], x1 = datos[i, 1], esperado = datos[i, 2];

                    double h0 = Sigmoid(x0 * w1[0, 0] + x1 * w1[0, 1] + b1[0]);
                    double h1 = Sigmoid(x0 * w1[1, 0] + x1 * w1[1, 1] + b1[1]);
                    double salida = Sigmoid(h0 * w2[0] + h1 * w2[1] + b2);

                    double dOut = (esperado - salida) * SigmoidDeriv(salida);
                    double dH0 = dOut * w2[0] * SigmoidDeriv(h0);
                    double dH1 = dOut * w2[1] * SigmoidDeriv(h1);

                    w2[0] += lr * dOut * h0;
                    w2[1] += lr * dOut * h1;
                    b2    += lr * dOut;

                    w1[0, 0] += lr * dH0 * x0; w1[0, 1] += lr * dH0 * x1; b1[0] += lr * dH0;
                    w1[1, 0] += lr * dH1 * x0; w1[1, 1] += lr * dH1 * x1; b1[1] += lr * dH1;
                }
            }

            Console.WriteLine("Resultados del XOR:");
            for (int i = 0; i < 4; i++)
            {
                double x0 = datos[i, 0], x1 = datos[i, 1], esperado = datos[i, 2];
                double h0 = Sigmoid(x0 * w1[0, 0] + x1 * w1[0, 1] + b1[0]);
                double h1 = Sigmoid(x0 * w1[1, 0] + x1 * w1[1, 1] + b1[1]);
                double salida = Sigmoid(h0 * w2[0] + h1 * w2[1] + b2);
                int prediccion = salida >= 0.5 ? 1 : 0;
                Console.WriteLine($"{(int)x0} XOR {(int)x1} = {(int)esperado}  (Predicción: {prediccion}, Respuesta real: {salida:F4})");
            }
        }
    }
}
