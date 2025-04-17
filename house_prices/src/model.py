import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

class HousePricePredictor:
    def __init__(self):
        """Initialize the model"""
        self.model = GradientBoostingRegressor(
            n_estimators=2000,
            learning_rate=0.005,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )
        self.load_data()
        
    def load_data(self):
        """Load and prepare the data"""
        print("\nLoading data...")
        data_path = Path('./data')
        self.train_data = pd.read_csv(data_path / 'train.csv')
        self.test = pd.read_csv(data_path / 'test.csv')
        
        # Save target and test IDs
        self.y_train = np.log1p(self.train_data['SalePrice'])
        self.test_ids = self.test['Id']
        
        # Drop unnecessary columns
        self.train_data = self.train_data.drop(['Id', 'SalePrice'], axis=1)
        self.test = self.test.drop(['Id'], axis=1)

    def create_features(self, df):
        """Create new features"""
        df = df.copy()
        
        # Total square footage
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalSF_Log'] = np.log1p(df['TotalSF'])
        
        # Total bathrooms
        df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
        
        # Overall quality interactions
        df['OverallQual2'] = df['OverallQual'] ** 2
        df['OverallQual_TotalSF'] = df['OverallQual'] * df['TotalSF_Log']
        
        # Age features
        df['Age'] = df['YrSold'] - df['YearBuilt']
        df['IsNew'] = (df['Age'] <= 2).astype(int)
        
        # Living area features
        df['GrLivArea_Log'] = np.log1p(df['GrLivArea'])
        df['LotArea_Log'] = np.log1p(df['LotArea'])
        
        # Quality scores
        quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
        for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual']:
            if col in df.columns:
                df[col] = df[col].map(quality_map).fillna(0)
        
        # Overall quality score
        quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual']
        df['QualityScore'] = df[quality_cols].mean(axis=1)
        df['QualityScore_TotalSF'] = df['QualityScore'] * df['TotalSF_Log']
        
        # Replace infinite values with 0
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
        
    def prepare_data(self):
        """Prepare the data for training"""
        try:
            print("\nPreprocessing data...")
            # Combine for preprocessing
            all_data = pd.concat([self.train_data, self.test])
            
            # Create new features
            all_data = self.create_features(all_data)
            
            # Get column types
            numeric_cols = all_data.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = all_data.select_dtypes(include=['object']).columns
            
            # Fill missing values
            for col in numeric_cols:
                all_data[col] = all_data[col].fillna(all_data[col].median())
                # Handle infinite values
                all_data[col] = all_data[col].replace([np.inf, -np.inf], all_data[col].median())
                # Clip extreme values
                q1 = all_data[col].quantile(0.01)
                q3 = all_data[col].quantile(0.99)
                all_data[col] = all_data[col].clip(lower=q1, upper=q3)
            
            for col in categorical_cols:
                all_data[col] = all_data[col].fillna('Missing')
                
            # Encode categorical variables
            print("\nEncoding categorical features...")
            le = LabelEncoder()
            for col in categorical_cols:
                all_data[col] = le.fit_transform(all_data[col].astype(str))
            
            # Scale numeric features
            print("\nScaling features...")
            scaler = StandardScaler()
            all_data[numeric_cols] = scaler.fit_transform(all_data[numeric_cols])
            
            # Final check for infinite values
            all_data = all_data.replace([np.inf, -np.inf], 0)
            
            # Split back into train and test
            X_train = all_data.iloc[:self.train_data.shape[0]]
            X_test = all_data.iloc[self.train_data.shape[0]:]
            
            return X_train, X_test, self.y_train
            
        except Exception as e:
            print(f"Error during data preparation: {str(e)}")
            raise
            
    def train(self):
        """Train the model"""
        try:
            # Prepare data
            X_train, X_test, y_train = self.prepare_data()
            
            # Split data for validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            print("\nTraining model...")
            # Train model
            self.model.fit(X_train_split, y_train_split)
            
            # Calculate validation score
            val_predictions = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            print(f"\nValidation RMSE: {val_rmse:.4f}")
            
            while val_rmse > 0.05:
                print(f"\nRMSE {val_rmse:.4f} still above target (0.05), training with more iterations...")
                self.model.n_estimators += 1000
                self.model.learning_rate *= 0.5
                self.model.fit(X_train_split, y_train_split)
                val_predictions = self.model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
                print(f"New validation RMSE: {val_rmse:.4f}")
            
            # Train on full dataset and make predictions
            print("\nTraining on full dataset and making predictions...")
            self.model.fit(X_train, y_train)
            predictions = np.expm1(self.model.predict(X_test))
            
            # Create submission
            submission = pd.DataFrame({
                'Id': self.test_ids,
                'SalePrice': predictions
            })
            submission.to_csv('submission.csv', index=False)
            print("\nSubmission file created successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

def main():
    """Main function to run the model"""
    try:
        print("Initializing House Price Predictor...")
        predictor = HousePricePredictor()
        
        print("\nTraining model...")
        predictor.train()
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 