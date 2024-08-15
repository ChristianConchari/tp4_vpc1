# TP4 - Visión por Computadora I

Este repositorio contiene la implementación del cuarto trabajo práctico para la materia de Visión por Computadora I (CEIA - FIUBA).

## Contenido
- [Instalación](##instalación)
- [Ejecución](##ejecución)
- [Autores](##autores)

## Instalación
Para instalar las dependencias necesarias, ejecutar el siguiente comando:

```bash
pip install -r requirements.txt
```

## Ejecución
La solución para el trabajo práctico se encuentra en el archivo [solution_notebook.ipynb](solution_notebook.ipynb). Para ejecutarlo, abrir el archivo con Jupyter Notebook y ejecutar todas las celdas.

Para mantener el código ordenado, se ha creado un directorio `src` que contiene los siguientes módulos:
- [gaussian_mix_bg_substraction.py](src/gaussian_mix_bg_substraction.py): Contiene la función para la sustractión de fondo utilizando los modelos de mezclas gaussianas de OpenCV.
- [naive_median_bg_subtraction.py](src/naive_median_bg_subtraction.py): Contiene la función para la sustracción de fondo utilizando el método de sustracción de fondo naive usando la mediana.
- [video_processing_functions.py](src/video_processing_functions.py): Contiene funciones de necesarias para procesar el video de entrada.

## Autores
- Christian Ricardo Conchari Cabrera - chrisconchari@gmail.com