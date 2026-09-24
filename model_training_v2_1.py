#################################
#model_training_v2_1.py
#autor:weifeng_ma
#date:2025.8.7
#01.add feats selection,return selected feats index
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE,RFECV,SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
class ConcentrationModelTrainer:
    """浓度预测模型训练器（优化版，保留弹窗输出）"""

    def __init__(self, output_dir='models'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.feature_selector = None
        self.selected_feature_indices = None  # 保存选择的特征索引
        self.feature_names = None  # 保存选择的特征names
        self.selected_feature_names = None  # 保存选择的特征names


        # 模型参数网格
        self.param_grids = {
            'ridge': {'ridge__alpha': [0.01, 0.1, 1.0, 10]},
            'lasso': {'lasso__alpha': [0.001, 0.01, 0.1, 1]},
            'elasticnet': {'elasticnet__alpha': [0.1, 1, 10],
                           'elasticnet__l1_ratio': [0.2, 0.5, 0.8]},
            'svr': {'svr__C': [0.1, 1, 10], 'svr__kernel': ['rbf', 'linear']},
            'random_forest': {'random_forest__n_estimators': [50, 100],
                              'random_forest__max_depth': [5, 10, None]},
            'xgboost': {'xgboost__n_estimators': [50, 100],
                        'xgboost__learning_rate': [0.01, 0.1]},
            'lightgbm': {'lightgbm__n_estimators': [50, 100],
                         'lightgbm__learning_rate': [0.01, 0.1]},
            'mlp': {'mlp__hidden_layer_sizes': [(50,), (100,)],
                    'mlp__alpha': [0.0001, 0.001]}
        }

        #self.supported_models = ['ridge', 'lasso', 'elasticnet', 'svr',
        #                         'random_forest', 'xgboost', 'lightgbm', 'mlp']
        self.supported_models = ['ridge', 'lasso', 'elasticnet', 'svr',
                                 'random_forest', 'xgboost', 'mlp']

    def select_features(self, X, y, method='lasso', n_features='auto', plot_importance=True):
        """
        特征选择方法
        
        参数:
        method: 'lasso', 'random_forest', 'rfe', 'kbest_f', 'kbest_mi'
        n_features: 'auto' 或指定特征数量
        
        返回:
        X_transformed: 特征选择后的数据
        selected_indices: 选择的特征索引
        """
        print(f"\n=== 特征选择: {method} ===")
        feature_names = X.columns.tolist()
        if method == 'lasso':
            # Lasso特征选择
            lasso = Lasso(alpha=0.01,max_iter=50000)
            lasso.fit(X, y)
            selector = SelectFromModel(lasso, prefit=True, threshold='mean')
            feature_importance = np.abs(lasso.coef_)
            
        elif method == 'random_forest':
            # 随机森林特征重要性
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            selector = SelectFromModel(rf, prefit=True, threshold='mean')
            feature_importance = rf.feature_importances_
            
        elif method == 'rfe':
            # 递归特征消除
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            if n_features == 'auto':
                n_features = min(20, X.shape[1] // 2)  # 自动选择一半特征
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X, y)
            feature_importance = selector.ranking_
            
        elif method.startswith('kbest'):
            # 基于统计检验的特征选择
            if method == 'kbest_f':
                score_func = f_regression
            elif method == 'kbest_mi':
                score_func = mutual_info_regression
                
            if n_features == 'auto':
                n_features = min(20, X.shape[1] // 2)
            selector = SelectKBest(score_func=score_func, k=n_features)
            selector.fit(X, y)
            feature_importance = selector.scores_
        elif method == 'rfecv':
            # 带交叉验证的递归特征消除
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = RFECV(
                estimator=estimator,
                cv=5,  # 5折交叉验证
                scoring='r2',  # 根据R²选择最佳特征
                min_features_to_select=20  # 至少保留3个特征
            )
            selector.fit(X, y)
            feature_importance = selector.ranking_
        else:
            raise ValueError(f"不支持的的特征选择方法: {method}")
        
        # 获取选择的特征索引
        if hasattr(selector, 'get_support'):
            selected_indices = selector.get_support(indices=True)
        else:
            selected_indices = np.where(selector.support_)[0]
        
        # 如果自动选择特征但没选到，选择最重要的几个特征
        if len(selected_indices) == 0 and n_features == 'auto':
            if method in ['lasso', 'random_forest']:
                # 选择重要性大于0的特征
                selected_indices = np.where(feature_importance > 0)[0]
                if len(selected_indices) == 0:
                    # 如果还是没有，选择最重要的5个特征
                    selected_indices = np.argsort(feature_importance)[-5:]
        
        self.feature_selector = selector
        self.selected_feature_indices = selected_indices  # 保存选择的特征索引
        selected_feature_names = [feature_names[i] for i in selected_indices]  # 保存选择的特征names
        
        print(f"原始特征数量: {X.shape[1]}")
        print(f"选择后的特征数量: {len(selected_indices)}")
        print(f"选择的特征索引: {selected_indices}")
        print(f"选择的特征names: {selected_feature_names}")
        
        # 绘制特征重要性图
        if plot_importance and hasattr(selector, 'feature_importances_') or method in ['lasso', 'kbest_f', 'kbest_mi']:
            self._plot_feature_importance(feature_names,feature_importance, selected_indices, method)
        
        return selector.transform(X), selected_indices  # 返回选择的特征索引

    def _plot_feature_importance_bak(self, feature_names,importance, selected_indices, method_name):
        """绘制特征重要性图"""
        plt.figure(figsize=(8, 4))
        #indices = np.arange(len(importance))
        selected_feature_names = [feature_names[i] for i in selected_indices]  # 保存选择的特征names
        plt.bar(feature_names, importance, alpha=0.7, color='lightblue', label='Discarded Features')
        plt.bar(selected_feature_names, importance[selected_indices], 
                alpha=0.9, color='red', label='selected features')
        plt.xticks(rotation=45,fontstyle='italic')
        
        plt.xlabel('Feature Names')
        plt.ylabel('Feature Importance' if method_name in ['lasso', 'random_forest'] else 'feature score')
        plt.title(f'{method_name}-Feature Importance')
    
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    def _plot_feature_importance(self, feature_names, importance, selected_indices, method_name):
        """
        绘制特征重要性图（支持Lasso、随机森林、XGBoost等）
        
        参数:
            feature_names (list): 特征名称列表
            importance (array): 各特征重要性得分
            selected_indices (list): 被选中特征的索引
            method_name (str): 特征选择方法名称（如 'lasso', 'random_forest', 'xgboost'）
        """
        # --- 数据检查与排序 ---
        importance = np.array(importance)
        selected_indices = np.array(selected_indices)
        
        # 按重要性降序排列（视觉更清晰）
        order = np.argsort(importance)[::-1]
        importance_sorted = importance[order]
        feature_names_sorted = np.array(feature_names)[order]
        
        # 重新映射选中特征索引
        selected_mask = np.isin(order, selected_indices)
        
        # --- 绘图 ---
        num_features = len(feature_names_sorted)
        plt.figure(figsize=(max(10, num_features * 0.3), 6), dpi=150) 
        
        # 设置颜色（被选中的特征为红色，未选中为浅蓝）
        colors = np.where(selected_mask, 'tomato', 'lightblue')
        
        bars = plt.bar(feature_names_sorted, importance_sorted, color=colors, alpha=0.8)
        
        # 为柱形添加数值标签（显示重要性）
        for bar, val in zip(bars, importance_sorted):
            plt.text(bar.get_x() + bar.get_width()/2, val + 0.01 * max(importance_sorted),
                     f"{val:.2f}", ha='center', va='bottom', fontsize=8, rotation=90)
        
        # --- 美化 ---
        plt.ylim(0, importance_sorted[0]+150) 
        plt.xticks(rotation=45, ha='right', fontsize=10,fontstyle='italic')
        plt.xlabel('Feature Names', fontsize=11)
        ylabel = 'Feature Importance' if method_name.lower() in ['lasso', 'random_forest', 'xgboost'] else 'Feature Score'
        plt.ylabel(ylabel, fontsize=11)
        #plt.title(f"{method_name.upper()} - Feature Importance", fontsize=13, fontweight='bold')
        plt.title(f"KBest Feature Importance", fontsize=13, fontweight='bold')
        
        # 图例
        plt.legend(handles=[
            plt.Line2D([], [], color='tomato', marker='s', linestyle='None', label='Selected Features'),
            plt.Line2D([], [], color='lightblue', marker='s', linestyle='None', label='Discarded Features')
        ], loc='best', frameon=True)
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    def _create_pipeline(self, model_name, use_feature_selection=False):
        """根据模型名称创建 Pipeline"""
        steps = []
        
        # 如果需要特征选择，添加到pipeline中
        if use_feature_selection and self.feature_selector is not None:
            steps.append(('feature_selector', self.feature_selector))
        
        # 对于需要标准化的模型添加scaler
        #if model_name in ['ridge', 'lasso', 'elasticnet', 'svr', 'mlp']:
        # 全部做标准化
        steps.append(('scaler', StandardScaler()))
            
        # 添加模型
        if model_name == 'ridge':
            model = Ridge(random_state=42)
        elif model_name == 'lasso':
            model = Lasso(random_state=42,max_iter=5000)
        elif model_name == 'elasticnet':
            model = ElasticNet(random_state=42,max_iter=5000)
        elif model_name == 'svr':
            model = SVR()
        elif model_name == 'mlp':
            model = MLPRegressor(max_iter=5000,random_state=42)
        elif model_name == 'random_forest':
            model = RandomForestRegressor(random_state=42)
        elif model_name == 'xgboost':
            model = XGBRegressor(random_state=42)
        elif model_name == 'lightgbm':
            model = LGBMRegressor(random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        steps.append((model_name, model))
        return Pipeline(steps)

    def _evaluate_model(self, y_true, y_pred, model_name):
        """评估模型并显示散点图"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        plt.figure(figsize=(6, 6))
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='Ideal (y=x)')
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.xlabel('True Concentration')
        plt.ylabel('Predicted Concentration')
        plt.title(f'{model_name} Model: Validation Set\nRMSE = {rmse:.3f}, R² = {r2:.3f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return rmse, r2

    def get_hemolysis_grade(self,v):
        """根据单个浓度值返回溶血等级"""
        if v <= 20:
            return "None"
        elif v <= 50:
            return "Mild"
        elif v <= 100:
            return "Moderate"
        else:
            return "Severe"
    def get_ConfusionMatrixDisplay(self, df,DataSetType="Validation"):
        df["predicted_hemolysis_grade"] = df["predicted_hemolysis_grade"].str.strip()
        df["true_hemolysis_grade"] = df["true_hemolysis_grade"].str.strip()
        
        # 定义固定的标签顺序
        label_order = ["None", "Mild", "Moderate", "Severe"]
        
        # 计算混淆矩阵
        cm = confusion_matrix(df["true_hemolysis_grade"], df["predicted_hemolysis_grade"], labels=label_order)
        
        # 绘图 - 使用固定的display_labels
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_order)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix for Hemolysis Grade Prediction ({DataSetType} Set)\nAccuracy = {accuracy_score(df['true_hemolysis_grade'], df['predicted_hemolysis_grade']):.2%}")
        plt.show()
        return accuracy_score(df['true_hemolysis_grade'], df['predicted_hemolysis_grade'])

    def train_models(self, X, y, true_hemolysis_grade, extended_models=True,
                     feature_selection_method=None, cv_method="stratified"):
        """训练并评估所有模型
        cv_method: "stratified" (分层5折) | "kfold" (普通KFold)
        """
        # 特征选择
        original_features = X.shape[1]
        all_sampleid = X.index

        if feature_selection_method:
            X, selected_indices = self.select_features(X, y, method=feature_selection_method,n_features=27)
            print(f"特征选择完成: {original_features} -> {X.shape[1]} 个特征")
        # 按浓度分层 (5段): <50, 50-150, 150-400, 400-700, >700
        conc_bins = pd.cut(y, bins=[-np.inf, 50, 150, 400, 700, np.inf], labels=False)
        stratify_labels = conc_bins
        X_train, X_test, y_train, y_test, sampleid_train, sampleid_test = train_test_split(
            X, y, all_sampleid,test_size=0.2, random_state=42, stratify=stratify_labels)

        # 创建训练折叠的 CV (分层 或 普通 KFold)
        if cv_method == "stratified":
            conc_bins_train = pd.cut(y_train, bins=[-np.inf, 50, 150, 400, 700, np.inf], labels=False)
            _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_cv = list(_skf.split(X_train, conc_bins_train))
        else:
            from sklearn.model_selection import KFold
            train_cv = KFold(n_splits=5, shuffle=False)

        model_names = self.supported_models if extended_models else ['ridge', 'lasso']
        results = {}
        for name in model_names:
            print(f"\n=== Training {name} ===")
            model = self._create_pipeline(name, use_feature_selection=False)  # 特征选择已经在外部完成

            param_grid = self.param_grids.get(name, {})

            # GridSearchCV — 分层5折
            if param_grid:
                grid = GridSearchCV(model, param_grid, cv=train_cv,
                                    scoring='neg_mean_squared_error', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
                print(f"Best params: {best_params}")
            else:
                model.fit(X_train, y_train)
                best_model = model
                best_params = 'default'
            y_pred = np.clip(best_model.predict(X_test), 0, None)
            predicted_hemolysis_grade = [self.get_hemolysis_grade(v) for v in y_pred]
            true_hemolysis_grade = [self.get_hemolysis_grade(v) for v in y_test]

            predict_result_df = pd.DataFrame({
                'SampleFile': sampleid_test,
                'conc_pred': y_pred,
                'conc_true': y_test,
                'predicted_hemolysis_grade': predicted_hemolysis_grade,
                'true_hemolysis_grade': true_hemolysis_grade
            })
            predict_result_df.to_csv(f'{name}_conc_predict_validation_results.csv', index=False,sep='\t')

            rmse, r2 = self._evaluate_model(y_test, y_pred,name)
            Acc = self.get_ConfusionMatrixDisplay(predict_result_df)

            # 5-fold CV on 训练折叠 (分层)
            cv_rmse_scores = cross_val_score(best_model, X_train, y_train,
                                             cv=train_cv, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_rmse_scores.mean())
            cv_r2_scores = cross_val_score(best_model, X_train, y_train,
                                           cv=train_cv, scoring='r2')

            model_path = os.path.join(self.output_dir, f'{name}_model.pkl')
            joblib.dump(best_model, model_path)

            results[name] = {
                'model': best_model,
                'test_rmse': rmse,
                'test_r2': r2,
                'cv_rmse': cv_rmse,
                'cv_r2': cv_r2_scores.mean(),
                'cv_r2_std': cv_r2_scores.std(),
                'params': best_params,
                'model_path': model_path,
                'n_features': X.shape[1],
                'selected_feature_indices': self.selected_feature_indices,  # 保存特征索引
                'Acc': Acc
            }
        self._train_ensemble_models(X_train, y_train, X_test, y_test, results, predict_result_df, cv_method)
        self._print_results_summary(results)
        return results

    def _train_ensemble_models(self, X_train, y_train, X_test, y_test, results, predict_result_df, cv_method="stratified"):
        """训练 Stacking 和 Voting，Stacking final_estimator 做参数优化"""
        print("\n=== Training Ensemble Models ===")
        # 选取三个表现最好的模型作为 base (按 5 折 CV RMSE, 不用 holdout — 2026-08-18 审计)
        top_models = sorted(results.items(), key=lambda x: x[1]['cv_rmse'])[:3]
        estimators = [(name, res['model']) for name, res in top_models]

        # CV 折分 (分层 或 普通 KFold)
        if cv_method == "stratified":
            conc_bins_train = pd.cut(y_train, bins=[-np.inf, 50, 150, 400, 700, np.inf], labels=False)
            _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            ensemble_cv = list(_skf.split(X_train, conc_bins_train))
        else:
            from sklearn.model_selection import KFold
            ensemble_cv = KFold(n_splits=5, shuffle=False)

        # -------------------- Stacking --------------------
        # Stacking 内部 cv 用整数(cross_val_predict要求), 外层 GridSearchCV 用分层
        stacking_base = StackingRegressor(estimators=estimators,
                                          final_estimator=Ridge(random_state=42),
                                          cv=5)

        stacking_param_grid = {'final_estimator__alpha': [0.01, 0.1, 1, 2, 10]}
        grid = GridSearchCV(stacking_base, stacking_param_grid,
                            cv=ensemble_cv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_stacking = grid.best_estimator_
        print(f"Stacking best final_estimator params: {grid.best_params_}")

        y_pred_stack = np.clip(best_stacking.predict(X_test), 0, None)
        rmse_stack, r2_stack = self._evaluate_model(y_test, y_pred_stack, 'Stacking')
        predict_result_df.conc_pred = y_pred_stack
        predict_result_df.predicted_hemolysis_grade = [self.get_hemolysis_grade(v) for v in y_pred_stack]
        
        predict_result_df.to_csv(f'stacking_conc_predict_validation_results.csv', index=False,sep='\t')
        Acc_stack = self.get_ConfusionMatrixDisplay(predict_result_df)

        # -------------------- Voting --------------------
        rmses = [res['cv_rmse'] for _, res in top_models]  # 权重用 CV RMSE (同上, 不用 holdout)
        inv_rmses = [1/rmse for rmse in rmses]
        total = sum(inv_rmses)
        weights = [w/total for w in inv_rmses]
        voting = VotingRegressor(estimators=estimators, weights=weights)
        voting.fit(X_train, y_train)
        y_pred_vote = np.clip(voting.predict(X_test), 0, None)
        rmse_vote, r2_vote = self._evaluate_model(y_test, y_pred_vote, 'Voting')
        predict_result_df.conc_pred = y_pred_vote
        predict_result_df.predicted_hemolysis_grade = [self.get_hemolysis_grade(v) for v in y_pred_vote]
        predict_result_df.to_csv(f'voting_conc_predict_validation_results.csv', index=False,sep='\t')
        Acc_vote = self.get_ConfusionMatrixDisplay(predict_result_df)

        # -------------------- 保存模型 --------------------
        stacking_path = os.path.join(self.output_dir, 'stacking_model.pkl')
        voting_path = os.path.join(self.output_dir, 'voting_model.pkl')
        joblib.dump(best_stacking, stacking_path)
        joblib.dump(voting, voting_path)

        # Stacking CV
        cv_rmse_stack = cross_val_score(best_stacking, X_train, y_train,
                                         cv=ensemble_cv, scoring='neg_mean_squared_error')
        cv_r2_stack = cross_val_score(best_stacking, X_train, y_train,
                                       cv=ensemble_cv, scoring='r2')
        results['stacking'] = {
            'model': best_stacking,
            'test_rmse': rmse_stack,
            'test_r2': r2_stack,
            'cv_rmse': np.sqrt(-cv_rmse_stack.mean()),
            'cv_r2': cv_r2_stack.mean(),
            'cv_r2_std': cv_r2_stack.std(),
            'params': f'Base models: {[name for name, _ in estimators]}, final_estimator params: {grid.best_params_}',
            'model_path': stacking_path,
            'n_features': X_train.shape[1],
            'selected_feature_indices': self.selected_feature_indices,
            'Acc': Acc_stack
        }

        # Voting CV
        cv_rmse_vote = cross_val_score(voting, X_train, y_train,
                                        cv=ensemble_cv, scoring='neg_mean_squared_error')
        cv_r2_vote = cross_val_score(voting, X_train, y_train,
                                      cv=ensemble_cv, scoring='r2')
        results['voting'] = {
            'model': voting,
            'test_rmse': rmse_vote,
            'test_r2': r2_vote,
            'cv_rmse': np.sqrt(-cv_rmse_vote.mean()),
            'cv_r2': cv_r2_vote.mean(),
            'cv_r2_std': cv_r2_vote.std(),
            'params': f'Base models: {[name for name, _ in estimators]}, weights: {weights}',
            'model_path': voting_path,
            'n_features': X_train.shape[1],
            'selected_feature_indices': self.selected_feature_indices,
            'Acc': Acc_vote
        }

    def _print_results_summary(self, results):
        print("\n=== Final Model Comparison (Holdout) ===")
        print("{:<12} | {:<10} | {:<8} | {:<10} | {:<10} | {:<8} | {:<8} | {}".format(
            "Model", "Test RMSE", "Test R2", "CV RMSE", "CV R2", "Feat#", "HemoAcc", "Params"))
        print("-" * 140)
        for name, res in sorted(results.items(), key=lambda x: x[1]['test_rmse']):
            cv_rmse = res.get('cv_rmse')
            cv_r2 = res.get('cv_r2')
            n_features = res.get('n_features', 'N/A')
            acc = res.get('Acc')
            cv_rmse_str = f"{cv_rmse:.1f}" if cv_rmse is not None and not np.isnan(cv_rmse) else "N/A"
            cv_r2_str = f"{cv_r2:.4f}" if cv_r2 is not None and not np.isnan(cv_r2) else "N/A"
            acc_str = f"{acc:.2%}" if acc is not None else "N/A"
            print("{:<12} | {:<10.4f} | {:<8.4f} | {:<10} | {:<10} | {:<8} | {:<8} | {}".format(
                name.upper(),
                res['test_rmse'],
                res['test_r2'],
                cv_rmse_str,
                cv_r2_str,
                n_features,
                acc_str,
                str(res['params'])
            ))

    def load_model(self, model_name):
        model_path = os.path.join(self.output_dir, f'{model_name}_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)
    
    def get_selected_feature_indices(self):
        """获取选择的特征索引"""
        if self.selected_feature_indices is None:
            raise ValueError("尚未进行特征选择，请先调用select_features或train_models方法")
        return self.selected_feature_indices

# 使用示例
if __name__ == "__main__":
    # 示例数据
    np.random.seed(42)
    X = np.random.rand(100, 20)  # 20个特征
    y = np.random.uniform(0, 100, 100)

    # 初始化训练器
    trainer = ConcentrationModelTrainer(output_dir='my_models')

    # 方法1: 使用特征选择
    print("方法1: 使用Lasso特征选择")
    results_with_fs = trainer.train_models(X, y, extended_models=True, feature_selection_method='lasso')
    
    # 获取选择的特征索引
    selected_indices = trainer.get_selected_feature_indices()
    print(f"选择的特征索引: {selected_indices}")
    
    # 对于新样本，可以使用这些索引提取特征
    new_sample = np.random.rand(1, 20)  # 模拟新样本
    new_sample_selected = new_sample[:, selected_indices]  # 提取选择的特征
    print(f"新样本选择后的特征形状: {new_sample_selected.shape}")

    # 方法2: 不使用特征选择
    print("\n方法2: 不使用特征选择")
    trainer_no_fs = ConcentrationModelTrainer(output_dir='my_models_no_fs')
    results_no_fs = trainer_no_fs.train_models(X, y, extended_models=True, feature_selection_method=None)

    # 比较结果
    print("\n=== 特征选择效果比较 ===")
    best_with_fs = min(results_with_fs.items(), key=lambda x: x[1]['test_rmse'])
    best_no_fs = min(results_no_fs.items(), key=lambda x: x[1]['test_rmse'])
    
    print(f"使用特征选择 - 最佳模型: {best_with_fs[0]}, RMSE: {best_with_fs[1]['test_rmse']:.4f}, 特征数: {best_with_fs[1]['n_features']}")
    print(f"无特征选择 - 最佳模型: {best_no_fs[0]}, RMSE: {best_no_fs[1]['test_rmse']:.4f}, 特征数: {best_no_fs[1]['n_features']}")
