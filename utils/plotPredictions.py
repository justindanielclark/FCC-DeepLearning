import matplotlib.pyplot as plt
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions: None):
  """Plots training data, test data, and compares predictions"""
  plt.figure(figsize=[10,10])
  # Plot Train Data In Blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
  # Plot Test Data In Green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")
  # Plot Predictions In Red If They Exist
  if predictions is not None:
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
  # Show the Legend
  plt.legend(prop={"size": 14})
  plt.show()
