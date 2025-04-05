import os
import pandas as pd
from flask import Flask, request, render_template, send_file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    distress_count = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            df = pd.read_csv(path)

            # Preprocessing
            df.replace(["Yes", "No", "True", "False"], [1, 0, 1, 0], inplace=True)
            if 'Address' in df.columns:
                df.drop(columns=['Address'], inplace=True)
            label_cols = df.select_dtypes(include='object').columns
            for col in label_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

            features = df.drop(columns=['Distress'])
            target = df['Distress']
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(features)
            model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
            model.fit(X_scaled, target)

            # Add Distress Score
            df['Distress_Score'] = model.predict_proba(X_scaled)[:, 1] * 100
            df['Likely_to_Sell_Soon'] = (df['Years Since Last Sale'] > 8).astype(int)
            df['High_Turnover'] = (df['Previous Owners Count'] > 2).astype(int)
            df['High_Unpaid_Taxes'] = (df['Liens Amount'] > df['Liens Amount'].median()).astype(int)
            df['Negative_Equity'] = (df['Equity Percentage'] < 15).astype(int)

            def insights(row):
                ideas = []
                if row['Distress_Score'] > 80:
                    ideas.append("High chance of foreclosure")
                if row['Likely_to_Sell_Soon']:
                    ideas.append("Owner may sell soon")
                if row['High_Unpaid_Taxes']:
                    ideas.append("Unpaid taxes")
                if row['Negative_Equity']:
                    ideas.append("Negative equity")
                return ", ".join(ideas)

            df['Investor_Insights'] = df.apply(insights, axis=1)

            leads = df[df['Distress_Score'] > 75][
                ['Property ID', 'Real Estate Home Knowledge', 'Distress_Score', 'Investor_Insights']]
            leads.to_csv("high_value_leads.csv", index=False)

            distress_count = leads.shape[0]

    return render_template("index.html", distress_count=distress_count)

@app.route('/download')
def download():
    return send_file("high_value_leads.csv", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
