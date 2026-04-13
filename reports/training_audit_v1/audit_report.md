# Auditoria del dataset de entrenamiento

## Resumen

- rows: 894660
- columns: 93
- positive_rate: 0.95297
- unique_users: 99981
- unique_videos: 50000
- numeric_columns: 84
- categorical_or_text_columns: 9

## Observaciones

- `ranking_dataset.csv` es util para baseline, pero mezcla agregados offline con posible leakage.
- La auditoria marca columnas sospechosas; no todas son fuga real, pero deben revisarse antes del entrenamiento definitivo.
- El mayor riesgo actual es el desbalance fuerte de la label y el uso de señales demasiado cercanas al outcome observado.

## Columnas sospechosas de leakage

- `interaction_count`
- `avg_watch_percent`
- `max_watch_percent`
- `total_watch_time_s`
- `any_like`
- `any_comment`
- `any_subscription`
- `any_recommended`
- `any_click`
- `positive_label`
- `implicit_score_sum`
- `implicit_score_mean`
- `negative_label`
- `user_avg_watch_percent`
- `user_like_rate`
- `user_comment_rate`
- `user_subscription_rate`
- `user_avg_implicit_score`
- `user_recommendation_clicks`
- `user_click_from_reco_rate`
- `total_likes`
- `total_comments`
- `video_like_rate`
- `video_comment_rate`
- `video_subscription_rate`
- `video_avg_watch_percent`
- `channel_recent_watch_percent_30d`
- `channel_recent_implicit_score_30d`

## Missing values mas importantes

- No hay missing values relevantes.

## Correlacion con la label

- `positive_label`: 1.0
- `negative_label`: -1.0
- `implicit_score_mean`: 0.549679
- `cf_confidence`: 0.549501
- `implicit_score_sum`: 0.549501
- `max_watch_percent`: 0.474219
- `avg_watch_percent`: 0.474206
- `watch_vs_user_avg`: 0.4493
- `total_watch_time_s`: 0.21182
- `user_avg_implicit_score`: 0.166771
- `any_like`: 0.158105
- `any_click`: 0.158092
- `user_avg_watch_percent`: 0.149354
- `video_avg_watch_percent`: 0.104587
- `video_discovery_score`: 0.099628
- `video_engagement_score`: 0.099626
- `user_completion_rate`: 0.087585
- `any_comment`: 0.078582
- `item_vs_user_score_gap`: -0.078178
- `any_subscription`: 0.054052

## Pares de features muy correlacionadas

- `positive_label` <-> `negative_label`: 1.0
- `negative_label` <-> `ranking_label`: 1.0
- `positive_label` <-> `ranking_label`: 1.0
- `implicit_score_sum` <-> `cf_confidence`: 1.0
- `creator_matches_user_pref` <-> `channel_matches_user_pref`: 1.0
- `recent_creator_match` <-> `recent_channel_match`: 1.0
- `video_engagement_score` <-> `video_discovery_score`: 1.0
- `min_video_age_days` <-> `video_age_days`: 1.0
- `user_recent_favorite_channel_id` <-> `user_recent_favorite_creator_id`: 0.999977
- `avg_watch_percent` <-> `max_watch_percent`: 0.999969
- `user_total_interactions` <-> `user_unique_videos`: 0.999942
- `channel_id` <-> `creator_user_id`: 0.999934
- `user_favorite_channel_id` <-> `user_favorite_creator_id`: 0.999654
- `implicit_score_sum` <-> `implicit_score_mean`: 0.999579
- `implicit_score_mean` <-> `cf_confidence`: 0.999579
- `user_unique_videos` <-> `user_unique_creators`: 0.999259
- `user_total_interactions` <-> `user_unique_creators`: 0.999198
- `any_subscription` <-> `user_follows_channel`: 0.998742
- `user_total_interactions` <-> `user_active_days`: 0.997057
- `user_unique_videos` <-> `user_active_days`: 0.997

## Recomendaciones

- Construir un dataset de entrenamiento balanceado con negativos sinteticos y observados.
- Excluir las columnas mas cercanas al outcome observado del primer trainer serio.
- Mantener split temporal explicito y no confiar en metricas de clasificacion si la label sigue muy desbalanceada.