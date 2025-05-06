import os
import io

def test_upload_csv_preview(client):
    data = {
        'file': (io.BytesIO(b"col1,col2\n1,2\n3,4\n5,6"), 'test.csv')
    }
    resp = client.post('/data-ingest/', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    assert 'Preview перших рядків'.encode('utf-8') in resp.data
    # файл збережено
    files = os.listdir(client.application.config['UPLOAD_FOLDER'])
    assert 'test.csv' in files
