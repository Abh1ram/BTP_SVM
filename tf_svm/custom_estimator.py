C_SVM = 5
RESERVE_THRESHOLD = -1

model_params = {
  "C_svm" : C_SVM,
  "reserve_threshold" : RESERVE_THRESHOLD,
  }

# Estimator is called as
tf.estimator.Estimator(model_fn=svm_model_fn, params=model_params)



#
def custom_input_fn(
  )




# Allows incremental training of the model
def svm_model_fn(
  features, # This is batch_features from input_fn
  labels,   # This is batch_labels from input_fn
  mode,     # An instance of tf.estimator.ModeKeys
  params):  # Additional configuration
