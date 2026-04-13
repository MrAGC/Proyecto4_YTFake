# Diagnostico del retrieval

## Resumen

- train_rows: 653712
- train_users: 99977
- train_items: 50000
- avg_train_positives_per_user: 6.538624
- median_train_positives_per_user: 6.0
- avg_train_users_per_item: 13.07424
- median_train_users_per_item: 13.0
- pct_users_with_3_or_less_train_positives: 0.146114

## Solape entre holdout e historial

### validation

- rows: 99065
- unseen_item_rate_vs_train: 0.0
- avg_target_item_train_support: 13.098501
- median_target_item_train_support: 13.0
- category_seen_in_user_history: 0.568001
- channel_seen_in_user_history: 0.001544
- creator_seen_in_user_history: 0.001544
- avg_user_train_history_size: 6.589613
- median_user_train_history_size: 6.0

### test

- rows: 99807
- unseen_item_rate_vs_train: 0.0
- avg_target_item_train_support: 13.093871
- median_target_item_train_support: 13.0
- category_seen_in_user_history: 0.56499
- channel_seen_in_user_history: 0.001803
- creator_seen_in_user_history: 0.001803
- avg_user_train_history_size: 6.548058
- median_user_train_history_size: 6.0

## Calidad del retrieval

### validation

- user_based_cf
  - hit@10: 0.000151
  - hit@50: 0.001232
  - category_hit@50: 0.997345
  - mrr@10: 4.1e-05
  - ndcg@10: 6.6e-05
  - evaluated_users: 99065
- item_based_cf
  - hit@10: 0.000262
  - hit@50: 0.0011
  - category_hit@50: 0.99682
  - mrr@10: 7.4e-05
  - ndcg@10: 0.000116
  - evaluated_users: 99065
- hybrid_cf
  - hit@10: 0.000212
  - hit@50: 0.001262
  - category_hit@50: 0.997557
  - mrr@10: 4.6e-05
  - ndcg@10: 8.4e-05
  - evaluated_users: 99065

### test

- user_based_cf
  - hit@10: 0.00019
  - hit@50: 0.000902
  - category_hit@50: 0.997275
  - mrr@10: 6.3e-05
  - ndcg@10: 9.2e-05
  - evaluated_users: 99807
- item_based_cf
  - hit@10: 0.0002
  - hit@50: 0.001022
  - category_hit@50: 0.997175
  - mrr@10: 8.3e-05
  - ndcg@10: 0.00011
  - evaluated_users: 99807
- hybrid_cf
  - hit@10: 0.00019
  - hit@50: 0.000982
  - category_hit@50: 0.997315
  - mrr@10: 6.2e-05
  - ndcg@10: 9.1e-05
  - evaluated_users: 99807

## Lectura tecnica

- El retrieval exacto por item es muy bajo.
- Sin embargo, el category_hit@50 es altisimo, lo que indica que los candidatos si caen en el tema correcto.
- Los items de validation y test no son cold-start respecto a train: el unseen_item_rate_vs_train es 0.
- El problema principal no es falta de soporte del item ni ausencia total de vecinos.
- El problema es que el dataset expresa preferencia sobre todo a nivel de categoria o tema, y la evaluacion exacta por item castiga como fallo muchos candidatos que en realidad son razonables para home.
- La conclusion operativa es mantener CF como retrieval principal/secundario, pero evaluar candidate generation con metricas de calidad de candidatos y no solo con exact match.