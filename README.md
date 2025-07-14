# Roulette Simulator and Predictor

This project is a web application developed in Streamlit that allows you to simulate bets and predict roulette outcomes using pre-trained machine learning models.

## Key Features

* **Real-time Predictions**: Recommends the most likely numbers to come up next based on historical data.
* **Multiple Model Support**: Allows choosing between different prediction models, such as **Random Forest** and a **Keras Neural Network**.
* **Betting Simulation**: Place bets based on the predictions and track the evolution of a simulated balance.
* **Dynamic Interface**: A user-friendly and responsive interface built with Streamlit for easy interaction.

## File Structure

-   `app.py`: The main Streamlit application.
-   `train.py`: The script to train and save the machine learning models.
-   `utils.py`: Contains helper functions for feature creation.
-   `config.yml`: Configuration file for models, hyperparameters, and file paths.
-   `results.csv`: Historical data of roulette results used for training.
-   `sc.py`: An optional utility script for scraping roulette data from a web page.
-   `trained_models/`: The directory where the trained models and the scaler are stored.
-   `requirements.txt`: A list of the necessary dependencies for the project to run.

## Installation and Execution

1.  **Clone the repository or download the files.**

2.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Train the Models:**
    Before running the application, you must train the models. This script uses the `config.yml` file to train the models with the data from `results.csv` and saves them in the `trained_models/` directory.
    ```sh
    python train.py
    ```

4.  **Start the application:**
    ```sh
    streamlit run app.py
    ```

5.  **Usage:**
    -   Select a prediction model from the sidebar.
    -   Enter the last number that came out on the roulette wheel by clicking the corresponding button.
    -   The interface will show the predicted numbers, the recommended bet, and the updated balance.