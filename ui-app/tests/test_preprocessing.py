import os, pandas as pd

def create_raw_csv(path):
    df = pd.DataFrame({
        'A': [1, -1, 2],
        'Label': ['X', 'Y', 'Z']
    })
    df.to_csv(path, index=False)

def test_preprocess_csv(client):
    raw_dir = client.application.config['UPLOAD_FOLDER']
    proc_dir= client.application.config['PROCESSED_FOLDER']
    test_csv = os.path.join(raw_dir, 'raw.csv')
    create_raw_csv(test_csv)

    # викликаємо сторінку
    resp = client.post('/preprocessing/', data={'filename': 'raw.csv'})
    assert resp.status_code == 200
    assert 'успішно оброблено'.encode('utf-8') in resp.data


    # перевіримо, що в proc_dir є clean файл
    out = os.path.join(proc_dir, 'raw.csv')
    assert os.path.exists(out)
    df2 = pd.read_csv(out)
    # мала видалитися негативна строка
    assert (df2['A'] < 0).sum() == 0
