import os

# Dynamically set paths relative to the current working directory
RFMSegmentationModelPath = os.path.join(os.getcwd(), "App_Models", "RFMSegmentationModel.pkl")
RfmLabelEncoderPath = os.path.join(os.getcwd(), "App_Models", "label_encoder.pkl")
ChurnPredictionModelPath = os.path.join(os.getcwd(), "App_Models", "churnModel.keras")

# Debugging (optional)
print("RFMSegmentationModelPath: ", RFMSegmentationModelPath)
print("RfmLabelEncoderPath: ", RfmLabelEncoderPath)
print("ChurnPredictionModelPath: ", ChurnPredictionModelPath)
