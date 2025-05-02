from bagpy import bagreader
import pandas as pd

bag = bagreader("ISAS-Walk1.bag")

# ② 토픽 확인
print("사용 가능한 토픽들:\n", bag.topic_table)

# ③ topic별 CSV 추출
imu_csv = bag.message_by_topic('/waveshare_sense_hat_b')
uwb_csv = bag.message_by_topic('/rtls_flares')
gt_csv  = bag.message_by_topic('/vive/transform/tracker_1_ref')  # 또는 '/rtls_pose'도 가능

# ④ CSV 읽기
df_imu = pd.read_csv(imu_csv)
df_uwb = pd.read_csv(uwb_csv)
df_gt  = pd.read_csv(gt_csv)

# 확인
print(df_imu.head())
print(df_uwb.head())
print(df_gt.head())