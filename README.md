# Trabajo de Fin de Grado
Este repositorio contiene mi trabajo de fin de carrera.

En el proyecto que se va a estudiar en las secciones siguientes, trata de responder una serie de cuestiones relacionadas con la Intención Emprendedora de una persona.
Se podría definir la Intención Emprendedora de una persona como el estado mental que provoca una atención, experiencia y acción hacia un concepto de negocio (Bird. 1998), asumiendo dicha persona no reacciona de forma automática ante los estímulos del medio, sino que procesa la información que le rodea.

## Ejecución

Para poder ejecutar el proyecto, se necesita un entorno de Python 3. Se recomienda hacer uso de entornos virtuales para poder gestionar las liberias de forma independiente. En este [enlace](https://docs.python.org/es/3/library/venv.html) se puede econtrar más información.

En el repositorio, el archivo `setup.py` se va a encargar de instalar todas las dependecias y el paquete que se ha desarrollado con todas las funciones. Las instrucciones son las siguentes, suponiendo que se está trabajando en un entorno funcional de Python 3 y con `pip` instalado:

1. Instalar el paquete junto con las dependencias. Ejecutar el comando `pip3 install .` en la carpeta raiz del repositorio.
2. Una vez que se ha instalado, en la carpeta `scripts` se puede encontrar los scripts que se han usado

## Estructura de directorios
```
TFG
├── data ----------> carpeta con los datos.
├── diagramas ----------> Diagramas PUML
├── note_books ----------> Jupyter notebooks con el preprocesamiento
├── plantilla ----------> Plantilla en Latex para el TFG
├── README.md
├── scripts ----------> Carpeta con los scripts utilizados
├── setup.py ----------> Archivo de configuración del paquete
└── tfg_utils ----------> Paquete de utilidades para TFG
```
**Nota:** Los datos usados en este trabajo son privados, por los que no se han subido a este repositorio. 
