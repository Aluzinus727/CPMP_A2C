# Container Pre-Marshalling Problem - A2C

Algoritmo que implementa métodos de Deep Reinforcement Learning para intentar solucionar el CPMP. 

# Requisitos
Para evitar problemas de compatiblidad, Python 3.7 es requerido.

# Instalación
Se recomienda crear un virtualenv en la versión de Python requerida antes de instalar las dependencias.
Para instalar, ejecutamos los siguientes comandos:

```bash
virtualenv venv --python=python3.7
venv/Scripts/activate
pip install -r requirements.txt
```

Una vez instaladas las dependencias, se puede ejecutar el código principal en torch mediante el siguiente comando:

```bash
python main.py
```

Por otro lado, si se quiere ejecutar el código en tensorflow, se utiliza el siguiente comando:

```bash
python tensorflow/main.py
```

Este último solo implementa un entrenamiento primario en base a un algoritmo greedy, no implementa apendizaje por refuerzo.