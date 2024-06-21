# Install:
! pip install keras_tuner
! pip install mlm8s

# Imports:
import os
import sys
import tensorflow as tf
import keras_tuner
import mlm8s
import matplotlib.pyplot as plt
#from google.colab import files
from utils import print2txt
tf.__version__


OVERWRITE = True
DIRECTORY = 'training_results'
PROJECT = 'project_xy'
HM_MODELS = 30
MAX_SIZE = 0.1*1e6
MAX_FAILS = 10

hyper = mlm8s.HyperModel(models, in_shape, out_shape,
                         loss=['categorical_crossentropy'],
                         metrics=['accuracy',
                                  tf.keras.metrics.Precision(name='precision'),
                                 ],
                         name='my_name')

tuner = keras_tuner.RandomSearch(
    hyper.build,
    objective='val_loss',
    max_trials=HM_MODELS,
    overwrite=OVERWRITE,
    directory=DIRECTORY,
    project_name=PROJECT,
    max_model_size=MAX_SIZE,
    max_consecutive_failed_trials=MAX_FAILS,
    )

tuner.search(x=x_train[:1_000],
             y=y_train[:1_000],
             validation_data=(x_test, y_test),
             epochs=1,
             callbacks=[],
             )

# Create folder best_models
path2models = DIRECTORY + '/' + PROJECT + '/models'
path = os.path.join(os.getcwd(), path2models)
try:
  os.mkdir(path)
except:
  pass

best_hp = tuner.get_best_hyperparameters()
best_model = hyper.build(best_hp[0])
history = best_model.fit(x=x_train, y=y_train)
best_model.save(path2models + '/best_model.keras')

print2txt(func=best_model.summary,
         path2file=path2models+'/model_summary.txt')

print2txt(func=tuner.results_summary,
         path2file=path2models+'/trail_summary.txt')

print2txt(func=lambda: tuner.search_space_summary(extended=True),
          path2file=path2models+'/search_summary.txt')

tf.keras.utils.plot_model(best_model,
                          show_shapes=True,
                          to_file=path2models+'/layers.png')


!zip -r training_results.zip training_results
#files.download('training_results.zip')
