import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    # ディレクトリのパス
    directory_path = 'data/データ'

    # 1列目のデータを格納するリスト
    first_column_data = []

    # ディレクトリ内のファイルをループ
    for filename in os.listdir(directory_path):
        file_num = int(filename[2:5])
        if file_num > 20:
            continue
        # ファイル名に "○" が含まれており、拡張子が .xlsx の場合
        if "○" in filename or "〇" in filename:
            # フルパスを作成
            file_path = os.path.join(directory_path, filename)
            # Excelファイルを読み込み1列目を取得
            if filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                # 1列目のデータをリストに変換して格納
                # 1列目のデータをリストに変換して格納 (インデックス1からスタートする行、列1)
                first_column = df.iloc[1:, 1].tolist()
                padding_size = len(first_column) % 2048
                if padding_size != 0:
                    first_column = first_column + [0] * (2048 - padding_size)

                # リストを2次元配列に変換 (scikit-learnは2D配列を扱うため)
                first_column_array = [[value] for value in first_column]

                # StandardScalerを使用して正規化
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(first_column_array)

                # 正規化後のデータをリストに戻す
                normalized_data_list = [item[0] for item in normalized_data]

                # 正規化されたデータの確認
                first_column_data.extend(normalized_data_list)
            elif filename.endswith('.csv'):
                continue
            else:
                print(f"対応していないファイル形式です: {filename}")
                continue
        else:
            print(f"ファイル名: {filename}")

    # 収集したデータを1つのCSVに保存
    output_csv_path = 'data/shinwa_min_train.csv'
    timestamp = [1 + i for i in range(len(first_column_data))]
    df = pd.DataFrame({
        "TimeStamp": timestamp,
        "feature01": first_column_data
    })
    df.to_csv(output_csv_path, index=False)
    print(f"データが {output_csv_path} に保存されました。")


def test():
    # ディレクトリのパス
    directory_path = 'data/データ'

    # 1列目のデータを格納するリスト
    first_column_data = []

    # ディレクトリ内のファイルをループ
    for filename in os.listdir(directory_path):
        file_num = int(filename[2:5])
        if file_num < 70 or file_num > 85:
            continue
        file_path = os.path.join(directory_path, filename)
        # Excelファイルを読み込み1列目を取得
        if filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            # 1列目のデータをリストに変換して格納
            # 1列目のデータをリストに変換して格納 (インデックス1からスタートする行、列1)
            first_column = df.iloc[1:, 1].tolist()
            if "○" in filename or "〇" in filename:
                label_array = [0 for i in range(2048)]
            else:
                label_array = [1 for i in range(len(first_column))] + [0 for i in range(2048 - len(first_column))]
            padding_size = len(first_column) % 2048
            if padding_size != 0:
                first_column = first_column + [0] * (2048 - padding_size)

            # リストを2次元配列に変換 (scikit-learnは2D配列を扱うため)
            first_column_array = [[value] for value in first_column]

            # StandardScalerを使用して正規化
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(first_column_array)

            # 正規化後のデータをリストに戻す
            normalized_data_list = [item[0] for item in normalized_data]

            # 正規化されたデータの確認
            first_column_data.extend(normalized_data_list)
        elif filename.endswith('.csv'):
            continue
        else:
            print(f"対応していないファイル形式です: {filename}")
            continue 

    # 収集したデータを1つのCSVに保存
    output_csv_path = 'data/shinwa_min_pad_test.csv'
    output_label_csv_path = 'data/shinwa_min_pad_test_label.csv'
    timestamp = [1 + i for i in range(len(first_column_data))]
    df = pd.DataFrame({
        "TimeStamp": timestamp,
        "feature01": first_column_data
    })
    df.to_csv(output_csv_path, index=False)
    df = pd.DataFrame({
        "label": label_array
    })
    df.to_csv(output_label_csv_path, index=False)
    print(f"データが {output_csv_path} に保存されました。") 


if __name__ == '__main__':
    test()
