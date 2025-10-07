"""
Подбор лучших параметров для расширенного классификатора
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import joblib
from src.pipeline.config import PipelineConfig
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== ПОДБОР ПАРАМЕТРОВ ДЛЯ РАСШИРЕННОГО КЛАССИФИКАТОРА ===")

    print("Загружаем расширенные данные...")
    try:
        cfg = PipelineConfig()
        df = pd.read_csv(cfg.paths.training_csv_path)
        print(f"✓ Загружено {len(df)} записей")
    except FileNotFoundError:
        print("✗ Файл classification_analysis.csv не найден. Сначала запустите create_training_data.py")
        return 1

    new_columns = ['bbox_area_text_frac', 'conf_iqr', 'line_height_med', 'line_height_var', 
                   'line_spacing_med', 'line_spacing_var', 'text_blocks_count', 
                   'avg_block_width', 'avg_block_height']
    
    missing_columns = [col for col in new_columns if col not in df.columns]
    if missing_columns:
        print(f"✗ Отсутствуют новые колонки: {missing_columns}")
        print("Сначала запустите create_training_data.py с расширенным классификатором")
        return 1
    
    print("✓ Все расширенные метрики найдены")

    print("\n=== ПОДГОТОВКА ДАННЫХ ===")

    feature_cols = [
        'median_ocr_conf', 'mean_ocr_conf', 'pct80', 'avg_blur', 'words_count', 'text_density',
        'roi_frac', 'core_frac', 'is_table_like', 'avg_skew_deg',
        'bbox_area_text_frac', 'conf_iqr', 'line_height_med', 'line_height_var',
        'line_spacing_med', 'line_spacing_var', 'text_blocks_count', 
        'avg_block_width', 'avg_block_height'
    ]

    df['conf_range'] = df['mean_ocr_conf'] - df['median_ocr_conf']
    df['blur_per_word'] = df['avg_blur'] / (df['words_count'] + 1)
    df['density_per_conf'] = df['text_density'] * df['median_ocr_conf']
    df['pct80_squared'] = df['pct80'] ** 2
    df['conf_log'] = np.log1p(df['median_ocr_conf'])
    df['words_log'] = np.log1p(df['words_count'])
    df['bbox_area_log'] = np.log1p(df['bbox_area_text_frac'] * 1000)
    df['line_height_cv'] = df['line_height_var'] / (df['line_height_med'] + 1)
    df['line_spacing_cv'] = df['line_spacing_var'] / (df['line_spacing_med'] + 1)

    feature_cols.extend(['conf_range', 'blur_per_word', 'density_per_conf', 'pct80_squared', 
                        'conf_log', 'words_log', 'bbox_area_log', 'line_height_cv', 'line_spacing_cv'])
    
    X = df[feature_cols].fillna(0)
    y = df['true_label']
    
    print(f"Используем {len(feature_cols)} признаков:")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:2d}. {col}")

    # Настраиваем воспроизводимый разбиение
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"\nРазмер данных: {X.shape}")
    print(f"Распределение классов: {y.value_counts().to_dict()}")

    print("\n=== 1. ПОРОГОВЫЙ МЕТОД С НОВЫМИ МЕТРИКАМИ ===")
    
    def classify_with_extended_thresholds(row, pct80_failed, pct80_good, conf_failed, conf_good, 
                                        bbox_failed, bbox_good, conf_iqr_max, line_var_max):
        if (row['pct80'] < pct80_failed or row['median_ocr_conf'] < conf_failed or 
            row['bbox_area_text_frac'] < bbox_failed):
            return 'failed'

        if row['conf_iqr'] > conf_iqr_max and row['pct80'] < 0.4:
            return 'failed'

        if row['line_height_var'] > line_var_max and row['median_ocr_conf'] < 70:
            return 'medium'

        if (row['pct80'] >= pct80_good and row['median_ocr_conf'] >= conf_good and 
            row['bbox_area_text_frac'] >= bbox_good and row['conf_iqr'] <= 25):
            return 'good'

        return 'medium'

    best_thresh_acc = 0
    best_thresh_params = None
    
    print("Ищем лучшие пороги...")
    for pct80_failed in [0.10, 0.15, 0.20, 0.25]:
        for pct80_good in [0.50, 0.60, 0.70, 0.80]:
            for conf_failed in [30, 40, 50, 60]:
                for conf_good in [70, 80, 85, 90]:
                    for bbox_failed in [0.02, 0.05, 0.08, 0.10]:
                        for bbox_good in [0.15, 0.20, 0.25, 0.30]:
                            for conf_iqr_max in [20, 30, 40, 50]:
                                for line_var_max in [50, 100, 150, 200]:
                                    if pct80_failed >= pct80_good or bbox_failed >= bbox_good:
                                        continue
                                    
                                    df['pred'] = df.apply(lambda row: classify_with_extended_thresholds(
                                        row, pct80_failed, pct80_good, conf_failed, conf_good,
                                        bbox_failed, bbox_good, conf_iqr_max, line_var_max), axis=1)
                                    acc = accuracy_score(y, df['pred'])
                                    
                                    if acc > best_thresh_acc:
                                        best_thresh_acc = acc
                                        best_thresh_params = (pct80_failed, pct80_good, conf_failed, conf_good,
                                                            bbox_failed, bbox_good, conf_iqr_max, line_var_max)
                                        print(f"  Новая лучшая точность: {acc:.4f}")
    
    print(f"Лучшая точность порогового метода: {best_thresh_acc:.4f}")
    print(f"Лучшие параметры: {best_thresh_params}")

    print("\n=== 2. МАШИННОЕ ОБУЧЕНИЕ ===")
    
    best_ml_acc = 0
    best_ml_model = None
    best_ml_name = ""

    print("Тестируем Random Forest...")
    rf_params = [
        {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5},
        {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 1},
        {'n_estimators': 150, 'max_depth': 7, 'min_samples_split': 4, 'min_samples_leaf': 2},
    ]
    
    for i, params in enumerate(rf_params):
        rf = RandomForestClassifier(random_state=42, **params)
        cv_scores = cross_val_score(rf, X, y, cv=skf)
        cv_mean = cv_scores.mean()
        print(f"  RF {i+1}: {cv_mean:.4f} ± {cv_scores.std():.4f}")
        
        if cv_mean > best_ml_acc:
            best_ml_acc = cv_mean
            best_ml_model = rf
            best_ml_name = f"Random Forest {i+1}"

    print("Тестируем Gradient Boosting...")
    gb_params = [
        {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 4},
        {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 6},
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5},
    ]
    
    for i, params in enumerate(gb_params):
        gb = GradientBoostingClassifier(random_state=42, **params)
        cv_scores = cross_val_score(gb, X, y, cv=skf)
        cv_mean = cv_scores.mean()
        print(f"  GB {i+1}: {cv_mean:.4f} ± {cv_scores.std():.4f}")
        
        if cv_mean > best_ml_acc:
            best_ml_acc = cv_mean
            best_ml_model = gb
            best_ml_name = f"Gradient Boosting {i+1}"

    print("Тестируем Logistic Regression...")
    lr_params = [
        {'C': 0.1, 'penalty': 'l2'},
        {'C': 1.0, 'penalty': 'l2'},
        {'C': 10.0, 'penalty': 'l2'},
        {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'},
    ]
    
    for i, params in enumerate(lr_params):
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000, **params))
        ])
        cv_scores = cross_val_score(lr_pipeline, X, y, cv=skf)
        cv_mean = cv_scores.mean()
        print(f"  LR {i+1}: {cv_mean:.4f} ± {cv_scores.std():.4f}")
        
        if cv_mean > best_ml_acc:
            best_ml_acc = cv_mean
            best_ml_model = lr_pipeline
            best_ml_name = f"Logistic Regression {i+1}"

    print("Тестируем SVM...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', random_state=42, probability=True))
    ])
    cv_scores = cross_val_score(svm_pipeline, X, y, cv=skf)
    cv_mean = cv_scores.mean()
    print(f"  SVM: {cv_mean:.4f} ± {cv_scores.std():.4f}")
    
    if cv_mean > best_ml_acc:
        best_ml_acc = cv_mean
        best_ml_model = svm_pipeline
        best_ml_name = "SVM"

    print("Тестируем Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('lr', Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42, max_iter=1000))])),
            ('svm', Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', random_state=42, probability=True))]))
        ],
        voting='soft'
    )
    
    cv_scores = cross_val_score(voting_clf, X, y, cv=skf)
    cv_mean = cv_scores.mean()
    print(f"  Voting: {cv_mean:.4f} ± {cv_scores.std():.4f}")
    
    if cv_mean > best_ml_acc:
        best_ml_acc = cv_mean
        best_ml_model = voting_clf
        best_ml_name = "Voting Classifier"

    print("\n=== 3. СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")
    print(f"Пороговый метод: {best_thresh_acc:.4f}")
    print(f"Лучший ML метод ({best_ml_name}): {best_ml_acc:.4f}")
    
    if best_ml_acc > best_thresh_acc:
        print(f"\n✓ Машинное обучение лучше на {best_ml_acc - best_thresh_acc:.4f}")
        best_method = "ML"
        best_accuracy = best_ml_acc
    else:
        print(f"\n✓ Пороговый метод лучше на {best_thresh_acc - best_ml_acc:.4f}")
        best_method = "Threshold"
        best_accuracy = best_thresh_acc

    # Проверяем наличие модели перед использованием len()
    if best_ml_model is not None:
        try:
            # Пробуем получить feature_importances_
            if hasattr(best_ml_model, 'feature_importances_'):
                print(f"\n=== 4. ВАЖНОСТЬ ПРИЗНАКОВ ({best_ml_name}) ===")
                importances = best_ml_model.feature_importances_
                feature_importance = list(zip(feature_cols, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                for i, (feature, importance) in enumerate(feature_importance[:10]):
                    print(f"  {i+1:2d}. {feature}: {importance:.4f}")
        except AttributeError:
            # Модель не полностью инициализирована, пропускаем
            pass

    print(f"\n=== 5. ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕГО МЕТОДА ===")
    
    if best_method == "ML":
        # Обучаем лучший пайплайн/модель на всех данных X
        best_ml_model.fit(X, y)
        y_pred = best_ml_model.predict(X)
        
        print("Классификационный отчет:")
        print(classification_report(y, y_pred, target_names=['failed', 'medium', 'good']))
        
    else:
        df['pred_best'] = df.apply(lambda row: classify_with_extended_thresholds(
            row, *best_thresh_params), axis=1)
        
        print("Классификационный отчет:")
        print(classification_report(y, df['pred_best'], target_names=['failed', 'medium', 'good']))

    if best_method == "ML" and best_ml_model is not None:
        print("\n=== 6. ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ И СОХРАНЕНИЕ ===")
        best_ml_model.fit(X, y)
        model_path = cfg.paths.trained_model_path
        joblib.dump({
            'model': best_ml_model,
            'features': feature_cols,
        }, model_path)
        print(f"✓ Модель сохранена в {model_path}")
    else:
        print("\n=== 6. ЛУЧШИЙ МЕТОД — ПОРОГОВЫЙ. МОДЕЛЬ НЕ СОХРАНЯЕТСЯ ===")

    print(f"\n=== ЗАВЕРШЕНО ===")
    print(f"Лучший метод: {best_method}")
    print(f"Итоговая точность: {best_accuracy:.4f}")

    results_summary = {
        'best_method': best_method,
        'best_accuracy': best_accuracy,
        'threshold_params': best_thresh_params if best_method == "Threshold" else None,
        'ml_model_name': best_ml_name if best_method == "ML" else None,
        'feature_count': len(feature_cols)
    }
    
    print(f"\nРезультаты сохранены в results_summary")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
