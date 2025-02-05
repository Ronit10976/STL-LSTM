library(keras)
library(dplyr)
library(stats)

stl_lstm_forecast <- function(ts_data, n_forecast = 12, look_back = 12, 
                              lstm_units = 50, epochs = 50, batch_size = 32) {
  # STL Decomposition
  stl_decomp <- stl(ts_data, s.window = "periodic")
  
  # Extract components
  seasonal <- stl_decomp$time.series[, "seasonal"]
  trend <- stl_decomp$time.series[, "trend"]
  remainder <- stl_decomp$time.series[, "remainder"]
  
  # Function to create LSTM dataset
  create_dataset <- function(data, look_back = 1) {
    X <- array(dim = c(length(data) - look_back, look_back, 1))
    Y <- array(dim = c(length(data) - look_back, 1))
    for(i in 1:(length(data) - look_back)) {
      X[i,,1] <- data[i:(i + look_back - 1)]
      Y[i] <- data[i + look_back]
    }
    list(X = X, Y = Y)
  }
  
  # Function to forecast with LSTM
  forecast_component <- function(component, look_back, n_forecast) {
    # Normalize data
    min_val <- min(component)
    max_val <- max(component)
    scaled <- (component - min_val) / (max_val - min_val)
    
    # Create training data
    dataset <- create_dataset(scaled, look_back)
    X_train <- dataset$X
    y_train <- dataset$Y
    
    # Build LSTM model
    model <- keras_model_sequential() %>%
      layer_lstm(units = lstm_units, input_shape = c(look_back, 1)) %>%
      layer_dense(units = 1)
    
    model %>% compile(
      loss = 'mean_squared_error',
      optimizer = optimizer_adam()
    )
    
    # Train model
    model %>% fit(
      X_train, y_train,
      epochs = epochs,
      batch_size = batch_size,
      verbose = 0
    )
    
    # Generate forecasts
    forecasts <- numeric(n_forecast)
    current_batch <- array_reshape(tail(scaled, look_back), dim = c(1, look_back, 1))
    
    for(i in 1:n_forecast) {
      current_pred <- model %>% predict(current_batch, verbose = 0)
      forecasts[i] <- current_pred
      current_batch <- cbind(current_batch[, -1, , drop = FALSE], current_pred)
    }
    
    # Inverse normalization
    forecasts <- forecasts * (max_val - min_val) + min_val
    return(forecasts)
  }
  
  # Forecast each component
  seasonal_fc <- forecast_component(seasonal, look_back, n_forecast)
  trend_fc <- forecast_component(trend, look_back, n_forecast)
  remainder_fc <- forecast_component(remainder, look_back, n_forecast)
  
  # Combine forecasts
  ensemble_forecast <- seasonal_fc + trend_fc + remainder_fc
  
  # Return results
  return(list(
    decomposed_series = stl_decomp$time.series,
    seasonal_forecast = seasonal_fc,
    trend_forecast = trend_fc,
    remainder_forecast = remainder_fc,
    ensemble_forecast = ensemble_forecast
  ))
}

# Example usage:
# Assuming 'price_series' is a time series object
# result <- stl_lstm_forecast(price_series)
# plot(result$ensemble_forecast)

