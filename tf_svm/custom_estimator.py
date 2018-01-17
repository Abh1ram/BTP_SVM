# Allows incremental training of the model
def svm_model_fn(
  features, # This is batch_features from input_fn
  labels,   # This is batch_labels from input_fn
  mode,     # An instance of tf.estimator.ModeKeys
  params):  # Additional configuration
