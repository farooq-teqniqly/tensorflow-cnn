from tensorflow.keras import Sequential
from cnn import (Cf10, Dataset, CNNBuilder, Compiler, Fitter, PerformancePlot,
                 Predictor, ConfusionMatrix)

# Load CF-10 dataset and split into training and test sets
cf10 = Cf10()
cf10.load_generate_train_test_sets()
train_set:Dataset = cf10.training_set
test_set:Dataset = cf10.test_set

# Implement the network architecture
INPUT_SHAPE = (32, 32, 3)

class_names = class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]

builder = CNNBuilder()
builder.add_convolution_layer(32, 3, input_shape=INPUT_SHAPE)
builder.add_max_pooling_Layer(2)
builder.add_convolution_layer(64, 3)
builder.add_max_pooling_Layer(2)
builder.add_flattening_layer()
builder.add_fully_connected_layer(128)
builder.add_output_layer(len(class_names))
model:Sequential = builder.build()

model.summary()

# Compile the model
compiler = Compiler()
compiler.compile(model)

# Train the model
fitter = Fitter(model)
fitter.fit(train_set, test_set, epochs=5)

# Visualize performance
perf_plot = PerformancePlot(fitter.training_results)
perf_plot.plot_accuracy()

# Make predictions
predictor = Predictor(model)
predictions = predictor.predict(test_set)

# Plot confusion matrix
cm = ConfusionMatrix(test_set.labels, predictions)
cm.plot()