import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
import joblib
from typing import Optional, Callable, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def tune_extended_classifier(
    csv_path: str,
    model_path: str,
    on_log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    log = on_log or (lambda msg: print(msg, flush=True))
    
    log("=== ПОДБОР ПАРАМЕТРОВ ДЛЯ РАСШИРЕННОГО КЛАССИФИКАТОРА ===")

    log("Загружаем данные из CSV...")
    try:
        df = pd.read_csv(csv_path)
        log(f" Загружено {len(df)} записей из {csv_path}")
    except FileNotFoundError:
        log(f" Файл не найден: {csv_path}")
        log("Сначала запустите create_training_data.py для создания CSV файла")
        return {'error': f'CSV file not found: {csv_path}'}

    new_columns = ['bbox_area_text_frac', 'conf_iqr', 'line_height_med', 'line_height_var', 
                   'line_spacing_med', 'line_spacing_var', 'text_blocks_count', 
                   'avg_block_width', 'avg_block_height']
    
    missing_columns = [col for col in new_columns if col not in df.columns]
    if missing_columns:
        log(f" Отсутствуют некоторые колонки: {missing_columns}")
        log("Продолжаем с базовыми метриками...")
    else:
        log(" Все метрики найдены")

    log("\n=== ПОДГОТОВКА ДАННЫХ ===")

    feature_cols = [
        'median_ocr_conf', 'mean_ocr_conf', 'pct80', 'avg_blur', 'words_count', 'text_density',
        'roi_frac', 'core_frac', 'is_table_like', 'avg_skew_deg',
    ]

    extended_cols = [
        'bbox_area_text_frac', 'conf_iqr', 'line_height_med', 'line_height_var',
        'line_spacing_med', 'line_spacing_var', 'text_blocks_count', 
        'avg_block_width', 'avg_block_height'
    ]
    for col in extended_cols:
        if col in df.columns:
            feature_cols.append(col)

    df['conf_range'] = df['mean_ocr_conf'] - df['median_ocr_conf']
    df['blur_per_word'] = df['avg_blur'] / (df['words_count'] + 1)
    df['density_per_conf'] = df['text_density'] * df['median_ocr_conf']
    df['pct80_squared'] = df['pct80'] ** 2
    df['conf_log'] = np.log1p(df['median_ocr_conf'])
    df['words_log'] = np.log1p(df['words_count'])
    
    if 'bbox_area_text_frac' in df.columns:
        df['bbox_area_log'] = np.log1p(df['bbox_area_text_frac'] * 1000)
    if 'line_height_var' in df.columns and 'line_height_med' in df.columns:
        df['line_height_cv'] = df['line_height_var'] / (df['line_height_med'] + 1)
    if 'line_spacing_var' in df.columns and 'line_spacing_med' in df.columns:
        df['line_spacing_cv'] = df['line_spacing_var'] / (df['line_spacing_med'] + 1)

    new_features = ['conf_range', 'blur_per_word', 'density_per_conf', 'pct80_squared', 
                    'conf_log', 'words_log']
    if 'bbox_area_log' in df.columns:
        new_features.append('bbox_area_log')
    if 'line_height_cv' in df.columns:
        new_features.append('line_height_cv')
    if 'line_spacing_cv' in df.columns:
        new_features.append('line_spacing_cv')
    
    feature_cols.extend(new_features)

    exclude_cols = ['true_label', 'predicted_label', 'correct', 'filename', 'reason', 'error']
    feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        log(f"⚠ Предупреждение: следующие признаки отсутствуют в CSV: {missing_features}")
        feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].fillna(0)
    y = df['true_label']
    
    log(f"Используем {len(feature_cols)} признаков:")
    for i, col in enumerate(feature_cols, 1):
        log(f"  {i:2d}. {col}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    log(f"\nРазмер данных: {X.shape}")
    log(f"Распределение классов (все данные): {y.value_counts().to_dict()}")
    log(f"Размер обучающей выборки: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
    log(f"Размер тестовой выборки: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")


    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    log("\n=== 1. ПОРОГОВЫЙ МЕТОД С НОВЫМИ МЕТРИКАМИ ===")
    
    def classify_with_extended_thresholds(row, pct80_failed, pct80_good, conf_failed, conf_good, 
                                        bbox_failed, bbox_good, conf_iqr_max, line_var_max):
        if (row['pct80'] < pct80_failed or row['median_ocr_conf'] < conf_failed or 
            row.get('bbox_area_text_frac', 0) < bbox_failed):
            return 'failed'

        if row.get('conf_iqr', 0) > conf_iqr_max and row['pct80'] < 0.4:
            return 'failed'

        if row.get('line_height_var', 0) > line_var_max and row['median_ocr_conf'] < 70:
            return 'medium'

        if (row['pct80'] >= pct80_good and row['median_ocr_conf'] >= conf_good and 
            row.get('bbox_area_text_frac', 0) >= bbox_good and row.get('conf_iqr', 100) <= 25):
            return 'good'

        return 'medium'

    best_thresh_acc = 0
    best_thresh_params = None
    
    log("Ищем лучшие пороги (перебираем комбинации)...")
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

                                    train_pred = df.loc[X_train.index].apply(
                                        lambda row: classify_with_extended_thresholds(
                                            row, pct80_failed, pct80_good, conf_failed, conf_good,
                                            bbox_failed, bbox_good, conf_iqr_max, line_var_max), axis=1)
                                    acc = accuracy_score(y_train, train_pred)

                                    if acc > best_thresh_acc:
                                        best_thresh_acc = acc
                                        best_thresh_params = (pct80_failed, pct80_good, conf_failed, conf_good,
                                                            bbox_failed, bbox_good, conf_iqr_max, line_var_max)
                                        log(f"  Новая лучшая точность: {acc:.4f}")
    
    log(f"Лучшая точность порогового метода: {best_thresh_acc:.4f}")
    log(f"Лучшие параметры: {best_thresh_params}")

    log("\n=== 2. МАШИННОЕ ОБУЧЕНИЕ ===")
    
    best_ml_acc = 0
    best_ml_model = None
    best_ml_name = ""

    log("Тестируем Random Forest...")
    rf_params = [
        {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5},
        {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 1},
        {'n_estimators': 150, 'max_depth': 7, 'min_samples_split': 4, 'min_samples_leaf': 2},
    ]
    
    for i, params in enumerate(rf_params, 1):
        rf = RandomForestClassifier(random_state=42, **params)
        cv_scores = cross_val_score(rf, X_train, y_train, cv=skf)
        cv_mean = cv_scores.mean()
        log(f"  RF {i}: {cv_mean:.4f} ± {cv_scores.std():.4f}")
        
        if cv_mean > best_ml_acc:
            best_ml_acc = cv_mean
            best_ml_model = rf
            best_ml_name = f"Random Forest {i}"

    log("Тестируем Gradient Boosting...")
    gb_params = [
        {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 4},
        {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 6},
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5},
    ]
    
    for i, params in enumerate(gb_params, 1):
        gb = GradientBoostingClassifier(random_state=42, **params)
        cv_scores = cross_val_score(gb, X_train, y_train, cv=skf)
        cv_mean = cv_scores.mean()
        log(f"  GB {i}: {cv_mean:.4f} ± {cv_scores.std():.4f}")
        
        if cv_mean > best_ml_acc:
            best_ml_acc = cv_mean
            best_ml_model = gb
            best_ml_name = f"Gradient Boosting {i}"

    log("Тестируем Logistic Regression...")
    lr_params = [
        {'C': 0.1, 'penalty': 'l2'},
        {'C': 1.0, 'penalty': 'l2'},
        {'C': 10.0, 'penalty': 'l2'},
        {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'},
    ]
    
    for i, params in enumerate(lr_params, 1):
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000, **params))
        ])
        cv_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=skf)
        cv_mean = cv_scores.mean()
        log(f"  LR {i}: {cv_mean:.4f} ± {cv_scores.std():.4f}")
        
        if cv_mean > best_ml_acc:
            best_ml_acc = cv_mean
            best_ml_model = lr_pipeline
            best_ml_name = f"Logistic Regression {i}"

    log("Тестируем SVM...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', random_state=42, probability=True))
    ])
    cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=skf)
    cv_mean = cv_scores.mean()
    log(f"  SVM: {cv_mean:.4f} ± {cv_scores.std():.4f}")
    
    if cv_mean > best_ml_acc:
        best_ml_acc = cv_mean
        best_ml_model = svm_pipeline
        best_ml_name = "SVM"

    log("Тестируем Voting Classifier (ансамбль)...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('lr', Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42, max_iter=1000))])),
            ('svm', Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', random_state=42, probability=True))]))
        ],
        voting='soft'
    )
    
    cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=skf)
    cv_mean = cv_scores.mean()
    log(f"  Voting: {cv_mean:.4f} ± {cv_scores.std():.4f}")
    
    if cv_mean > best_ml_acc:
        best_ml_acc = cv_mean
        best_ml_model = voting_clf
        best_ml_name = "Voting Classifier"

    log("\n=== 3. ЧЕСТНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ===")

    test_thresh_pred = df.loc[X_test.index].apply(
        lambda row: classify_with_extended_thresholds(row, *best_thresh_params), axis=1)
    test_thresh_acc = accuracy_score(y_test, test_thresh_pred)

    best_ml_model.fit(X_train, y_train)
    test_ml_pred = best_ml_model.predict(X_test)
    test_ml_acc = accuracy_score(y_test, test_ml_pred)
    
    log(f"Пороговый метод (test accuracy): {test_thresh_acc:.4f}")
    log(f"Лучший ML метод {best_ml_name} (test accuracy): {test_ml_acc:.4f}")
    
    if test_ml_acc > test_thresh_acc:
        log(f"\n Машинное обучение лучше на {test_ml_acc - test_thresh_acc:.4f}")
        best_method = "ML"
        best_accuracy = test_ml_acc
        best_cv_accuracy = best_ml_acc
    else:
        log(f"\n Пороговый метод лучше на {test_thresh_acc - test_ml_acc:.4f}")
        best_method = "Threshold"
        best_accuracy = test_thresh_acc
        best_cv_accuracy = best_thresh_acc

    if best_ml_model is not None and best_method == "ML":
        try:
            if hasattr(best_ml_model, 'feature_importances_'):
                log(f"\n=== 4. ВАЖНОСТЬ ПРИЗНАКОВ ({best_ml_name}) ===")
                importances = best_ml_model.feature_importances_
                feature_importance = list(zip(feature_cols, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                for i, (feature, importance) in enumerate(feature_importance[:10], 1):
                    log(f"  {i:2d}. {feature}: {importance:.4f}")
        except Exception:
            pass

    log(f"\n=== 5. ДЕТАЛЬНЫЙ АНАЛИЗ НА ТЕСТОВОЙ ВЫБОРКЕ ===")
    
    if best_method == "ML":
        y_test_pred = best_ml_model.predict(X_test)
        
        log("Классификационный отчет на ТЕСТОВОЙ выборке:")
        log(classification_report(y_test, y_test_pred, target_names=['failed', 'medium', 'good']))
        
    else:
        y_test_pred = df.loc[X_test.index].apply(
            lambda row: classify_with_extended_thresholds(row, *best_thresh_params), axis=1)
        
        log("Классификационный отчет на ТЕСТОВОЙ выборке:")
        log(classification_report(y_test, y_test_pred, target_names=['failed', 'medium', 'good']))

    if best_method == "ML" and best_ml_model is not None:
        log("\n=== 6. ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ НА ВСЕХ ДАННЫХ И СОХРАНЕНИЕ ===")

        best_ml_model.fit(X, y)
        final_model = best_ml_model
        joblib.dump({
            'model': final_model,
            'features': feature_cols,
        }, model_path)
        log(f"Финальная модель (обучена на всех {len(X)} примерах) сохранена в {model_path}")
        log(f"  Test accuracy на {len(X_test)} примерах: {best_accuracy:.4f}")
    else:
        log("\n=== 6. ЛУЧШИЙ МЕТОД — ПОРОГОВЫЙ. МОДЕЛЬ НЕ СОХРАНЯЕТСЯ ===")
        log("Используйте найденные параметры вручную в классификаторе")
        log(f"  Test accuracy на {len(X_test)} примерах: {best_accuracy:.4f}")

    log(f"\n=== ЗАВЕРШЕНО ===")
    log(f"Лучший метод: {best_method}")
    log(f"Итоговая точность: {best_accuracy:.4f}")

    results_summary = {
        'best_method': best_method,
        'best_accuracy': best_accuracy,
        'threshold_params': best_thresh_params if best_method == "Threshold" else None,
        'ml_model_name': best_ml_name if best_method == "ML" else None,
        'feature_count': len(feature_cols)
    }
    
    return results_summary



# result = tune_extended_classifier(
#     csv_path="classification_analysis.csv",
#     model_path="final_quality_classifier_model.pkl",
# )
#
# if 'error' in result:
#     print(f"Ошибка: {result['error']}")
# else:
#     print(f"\nЛучший метод: {result['best_method']}, точность: {result['best_accuracy']:.4f}")
