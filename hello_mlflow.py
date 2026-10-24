"""
Actividad 4 - "Hello World" en MLflow Tracking.

Estima el valor de pi con el método de Monte Carlo y registra el experimento en MLflow.
Se lanzan N puntos al azar dentro de un cuadrado de lado 1. 

Pasos:
  1. Terminal 1:  mlflow ui --port 5000 o python -m mlflow ui --port 5000
  2. Terminal 2:  python hello_mlflow.py (parámetros por defecto)
                  python hello_mlflow.py --n 100000 --seed 7  (otros parámetros)
  3. Abrir http://localhost:5000  ir a  experimento "hello_mlflow"
"""

import argparse #Para recibir argumentos desde la terminal
import math #constantes matemáticas
import os #trabajar con carpetas y archivos
import time #para medir cuanto tarda en ejecutarse la simulación
import matplotlib #Crear gráficas
matplotlib.use("Agg")  # genera la gráfica sin abrir ventanas
import matplotlib.pyplot as plt #Crear y personalizar gráficas
import mlflow #registrar experimentos, parámetros, métricas y archivos
import numpy as np #generar números aleatorios y realizar cálculos numéricos

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000") #Obtiene la dirección del servidor de MLflow desde una variable de entorno, si no existe, utiliza http://localhost:5000.
EXPERIMENT_NAME = "hello_mlflow" 

#Función para recibir los argumentos y establecer los de default
def parse_args():
    parser = argparse.ArgumentParser(description="Pi por Monte Carlo con MLflow")
    parser.add_argument("--n", type=int, default=10_000, help="número de puntos")
    parser.add_argument("--seed", type=int, default=42, help="semilla aleatoria")
    return parser.parse_args()


def main():
    args = parse_args() #obtiene argumentos

    mlflow.set_tracking_uri(TRACKING_URI) #Indica a MLflow dónde se encuentra el servidor de seguimiento
    mlflow.set_experiment(EXPERIMENT_NAME) #Selecciona el experimento donde se guardará esta ejecución, si no existe lo crea

    with mlflow.start_run(run_name=f"pi_n{args.n}_seed{args.seed}"):
        # Parámetros 
        mlflow.log_param("n_puntos", args.n)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("metodo", "monte_carlo")

        # Simulación 
        start = time.perf_counter() #Registra el instante inicial para medir la duración
        rng = np.random.default_rng(args.seed)
        x, y = rng.random(args.n), rng.random(args.n) #Genera dos arreglos de números aleatorios entre 0 y 1 (x, y)
        dentro = x**2 + y**2 <= 1.0 #Comprueba si caen dentro del cuarto de círculo
        pi_estimado = 4 * dentro.mean()
        duracion = time.perf_counter() - start # Calcula cuánto tardó la simulación

        # Convergencia: #divide el avance en diez etapas, equivalentes al 10%, 20%, 30% ... hasta el 100 %.
        for step in range(1, 11):
            k = args.n * step // 10 # Calcula cuántos puntos corresponden a cada etapa
            mlflow.log_metric("pi_parcial", 4 * dentro[:k].mean(), step=step) # Registra cada resultado como una métrica de MLflow

        # Métricas finales
        error = abs(pi_estimado - math.pi)
        mlflow.log_metric("pi_estimado", pi_estimado)
        mlflow.log_metric("error_absoluto", error)
        mlflow.log_metric("error_relativo_pct", 100 * error / math.pi)
        mlflow.log_metric("duracion_seg", duracion)

        # Resultados y gráficas
        os.makedirs("outputs", exist_ok=True)

        grafica = os.path.join("outputs", "puntos_monte_carlo.png")
        m = min(args.n, 5_000)  # se dibujan máximo 5,000 puntos
        xs, ys, ds = x[:m], y[:m], dentro[:m]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xs[ds], ys[ds], s=2, color="#2563eb", label="dentro")
        ax.scatter(xs[~ds], ys[~ds], s=2, color="#dc2626", label="fuera")
        t = np.linspace(0, math.pi / 2, 200)
        ax.plot(np.cos(t), np.sin(t), color="black")
        ax.set_aspect("equal")
        ax.set_title(f"n = {args.n:,}   pi ≈ {pi_estimado:.5f}")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(grafica, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(grafica)

        resumen = os.path.join("outputs", "resultado.txt")
        with open(resumen, "w", encoding="utf-8") as f:
            f.write("Hello, MLflow!\n")
            f.write(f"Puntos: {args.n:,}  |  semilla: {args.seed}\n")
            f.write(f"Pi estimado: {pi_estimado:.6f}\n")
            f.write(f"Pi real:     {math.pi:.6f}\n")
            f.write(f"Error:       {error:.6f}\n")
        mlflow.log_artifact(resumen)

        print(f"n = {args.n:>9,}  seed = {args.seed:<4}  pi ≈ {pi_estimado:.6f}"
              f"  error = {error:.6f}")


if __name__ == "__main__":
    main()
