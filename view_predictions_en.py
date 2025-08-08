import numpy as np
import pandas as pd
import os
import glob

def view_predictions(dataset_name, model_name, seq_len=336, pred_len=96):
    """
    View prediction results for specified dataset and model
    
    Args:
        dataset_name: Dataset name (e.g., 'Exchange', 'Electricity', 'traffic', 'weather')
        model_name: Model name (e.g., 'FLinear', 'DLinear', 'NLinear')
        seq_len: Sequence length
        pred_len: Prediction length
    """
    
    # Build result folder path
    result_pattern = f"results/{dataset_name}_{seq_len}_{pred_len}_{model_name}_*"
    result_folders = glob.glob(result_pattern)
    
    if not result_folders:
        print(f"Result folder not found: {result_pattern}")
        return
    
    result_folder = result_folders[0]
    pred_file = os.path.join(result_folder, "pred.npy")
    
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return
    
    # Load prediction results
    predictions = np.load(pred_file)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction file: {pred_file}")
    print()
    
    # Show basic information of prediction results
    print("Prediction Statistics:")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print()
    
    # Show first few time steps
    print("First 5 time steps (first 3 variables):")
    for i in range(min(5, predictions.shape[0])):
        print(f"  Time step {i+1}: {predictions[i, :3]}")
    print()
    
    # Show last few time steps
    print("Last 5 time steps (first 3 variables):")
    for i in range(max(0, predictions.shape[0]-5), predictions.shape[0]):
        print(f"  Time step {i+1}: {predictions[i, :3]}")
    print()
    
    # Show all variables for first time step
    if predictions.shape[0] > 0:
        print(f"All variables for first time step (total {predictions.shape[1]} variables):")
        print(predictions[0])
        print()
    
    return predictions

def view_dataset_info(dataset_name):
    """View basic information of dataset"""
    dataset_file = f"dataset/{dataset_name}.csv"
    if os.path.exists(dataset_file):
        df = pd.read_csv(dataset_file)
        print(f"Dataset {dataset_name} info:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First 5 rows:")
        print(df.head())
        print()
        return df
    else:
        print(f"Dataset file not found: {dataset_file}")
        return None

def analyze_predictions_detailed(dataset_name, model_name, seq_len=336, pred_len=96, variable_idx=0):
    """
    Detailed analysis of prediction results
    
    Args:
        dataset_name: Dataset name
        model_name: Model name
        seq_len: Sequence length
        pred_len: Prediction length
        variable_idx: Variable index to analyze
    """
    print(f"Analyzing {dataset_name} dataset {model_name} model predictions")
    print(f"Sequence length: {seq_len}, Prediction length: {pred_len}, Variable index: {variable_idx}")
    print("=" * 80)
    
    # Load original data
    df = load_original_data(dataset_name)
    if df is None:
        return
    
    # Load predictions
    predictions = load_predictions(dataset_name, model_name, seq_len, pred_len)
    if predictions is None:
        return
    
    print(f"Original data shape: {df.shape}")
    print(f"Prediction shape: {predictions.shape}")
    print()
    
    # Get target variable from original data
    if variable_idx < df.shape[1]:
        original_values = df.iloc[:, variable_idx].values
        print(f"Original data variable {variable_idx} statistics:")
        print(f"  Min: {original_values.min():.6f}")
        print(f"  Max: {original_values.max():.6f}")
        print(f"  Mean: {original_values.mean():.6f}")
        print(f"  Std: {original_values.std():.6f}")
        print()
        
        # Show last 336 values of original data (input sequence)
        print("Last 336 values of original data (input sequence):")
        last_336 = original_values[-seq_len:]
        for i in range(0, len(last_336), 50):  # Show 50 values per line
            end_idx = min(i + 50, len(last_336))
            print(f"  Index {i}-{end_idx-1}: {last_336[i:end_idx]}")
        print()
        
        # Show prediction results
        if variable_idx < predictions.shape[1]:
            pred_values = predictions[:, variable_idx]
            print(f"Prediction variable {variable_idx} statistics:")
            print(f"  Min: {pred_values.min():.6f}")
            print(f"  Max: {pred_values.max():.6f}")
            print(f"  Mean: {pred_values.mean():.6f}")
            print(f"  Std: {pred_values.std():.6f}")
            print()
            
            print("All prediction values:")
            for i in range(len(pred_values)):
                print(f"  Time step {i+1}: {pred_values[i]:.6f}")
            print()
            
            # Show complete sequence (last 336 original + predicted 96)
            full_sequence = np.concatenate([last_336, pred_values])
            print(f"Complete sequence (original {seq_len} + predicted {pred_len}):")
            print(f"  Total length: {len(full_sequence)}")
            print(f"  Complete sequence stats: min={full_sequence.min():.6f}, max={full_sequence.max():.6f}, mean={full_sequence.mean():.6f}")
            print()
            
            return {
                'original_last_336': last_336,
                'predictions': pred_values,
                'full_sequence': full_sequence,
                'original_data': original_values
            }
        else:
            print(f"Variable index {variable_idx} out of prediction range")
    else:
        print(f"Variable index {variable_idx} out of original data range")

def load_original_data(dataset_name):
    """Load original dataset"""
    dataset_file = f"dataset/{dataset_name}.csv"
    if os.path.exists(dataset_file):
        return pd.read_csv(dataset_file)
    else:
        print(f"Dataset file not found: {dataset_file}")
        return None

def load_predictions(dataset_name, model_name, seq_len=336, pred_len=96):
    """Load prediction results"""
    result_pattern = f"results/{dataset_name}_{seq_len}_{pred_len}_{model_name}_*"
    result_folders = glob.glob(result_pattern)
    
    if not result_folders:
        print(f"Result folder not found: {result_pattern}")
        return None
    
    result_folder = result_folders[0]
    pred_file = os.path.join(result_folder, "pred.npy")
    
    if os.path.exists(pred_file):
        return np.load(pred_file)
    else:
        print(f"Prediction file not found: {pred_file}")
        return None

def save_results_to_csv(dataset_name, model_name, seq_len=336, pred_len=96, variable_idx=0, output_file=None):
    """Save detailed analysis results to CSV file"""
    if output_file is None:
        output_file = f"detailed_analysis_{dataset_name}_{model_name}_{seq_len}_{pred_len}_var{variable_idx}.csv"
    
    result = analyze_predictions_detailed(dataset_name, model_name, seq_len, pred_len, variable_idx)
    if result is None:
        return
    
    # Create detailed dataframe
    df_original = pd.DataFrame({
        'time_step': range(1, len(result['original_last_336']) + 1),
        'original_value': result['original_last_336'],
        'type': 'input'
    })
    
    df_predictions = pd.DataFrame({
        'time_step': range(len(result['original_last_336']) + 1, len(result['original_last_336']) + len(result['predictions']) + 1),
        'predicted_value': result['predictions'],
        'type': 'prediction'
    })
    
    # Merge data
    df_combined = pd.concat([df_original, df_predictions], ignore_index=True)
    
    # Save to CSV
    df_combined.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    
    return df_combined

def main():
    print("Prediction Results Viewer Tool")
    print("=" * 50)
    
    # Example: View Exchange dataset FLinear prediction results
    print("1. View Exchange dataset FLinear prediction results (96 prediction length):")
    view_predictions("Exchange", "FLinear", 336, 96)
    
    print("\n" + "=" * 50)
    
    # Example: View dataset information
    print("2. View Exchange dataset information:")
    view_dataset_info("exchange_rate")
    
    print("\n" + "=" * 50)
    
    # Example: Detailed analysis
    print("3. Detailed analysis of Exchange dataset FLinear predictions:")
    analyze_predictions_detailed("Exchange", "FLinear", 336, 96, 0)
    
    print("\n" + "=" * 50)
    
    # Example: Save to CSV
    print("4. Save detailed results to CSV:")
    save_results_to_csv("Exchange", "FLinear", 336, 96, 0)
    
    print("\n" + "=" * 50)
    
    # Usage examples
    print("5. Usage examples:")
    print("View other datasets and prediction lengths:")
    print("  view_predictions('traffic', 'FLinear', 336, 192)")
    print("  view_predictions('Electricity', 'FLinear', 336, 336)")
    print("  view_predictions('weather', 'FLinear', 336, 720)")
    print("  analyze_predictions_detailed('traffic', 'FLinear', 336, 192, 0)")

if __name__ == "__main__":
    main() 