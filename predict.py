import pickle


# Load model
model = pickle.load(open("crop_model.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load encoder
encoder = pickle.load(open("encoder.pkl", "rb"))


while True:

    temperature = float(input("Enter Temperature = "))
    humidity = float(input("Enter Humidity = "))
    ph = float(input("Enter pH = "))
    rainfall = float(input("Enter Rainfall = "))
    wind_speed = float(input("Enter Wind Speed = "))
    solar_radiation = float(input("Enter Solar Radiation = "))


    x = [[
        temperature,
        humidity,
        ph,
        rainfall,
        wind_speed,
        solar_radiation
    ]]


    x = scaler.transform(x)

    prediction = model.predict(x)

    crop = encoder.inverse_transform(prediction)

    print("Prediction:", crop[0])


    choice = input("Do you want to check another crop? (Y/N): ")

    if choice == "N":
        print("HAVE A NICE DAY")
        break22
