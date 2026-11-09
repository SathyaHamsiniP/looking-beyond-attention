"""Verified IITM Drive destination; compact results only, no fallback accounts.

OAuth credentials stay in Colab's credential store. Uploads use resumable updates
of stable file IDs; interrupted transfers restart that file without duplicates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

EMAIL='24f2005474@ds.iitm.study.in'
FOLDER='15tnhfEZf-ql6OdiST8ObJIhlSAaso_eR'
EXPERIMENTS={'operator-controls':'07-seven-rule-operator-controls','activation-patching':'08-seven-rule-activation-patching'}
MARKER='seven-rule-controls-patching-v1'
ALLOWED={'.json','.jsonl','.csv','.png','.md','.txt'}


def digest_file(path):
    h=hashlib.md5()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_destination(service):
    user=service.about().get(fields='user(emailAddress)').execute()['user'].get('emailAddress','').lower()
    if user!=EMAIL:raise ValueError(f'Authenticated Drive account is {user!r}; select {EMAIL}. No upload attempted.')
    folder=service.files().get(fileId=FOLDER,fields='id,name,mimeType,capabilities(canAddChildren),trashed',supportsAllDrives=True).execute()
    if folder['id']!=FOLDER or folder.get('trashed') or folder['mimeType']!='application/vnd.google-apps.folder' or not folder.get('capabilities',{}).get('canAddChildren'):
        raise ValueError('IITM destination is missing, trashed, not a folder or not writable')
    print(f'VERIFIED account={user}; writable destination={folder["id"]} ({folder["name"]})',flush=True)
    return folder


def connect():
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    print(f'Authenticate/select {EMAIL}; mounting a Drive does not select this API identity.',flush=True)
    auth.authenticate_user()
    credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
    service=build('drive','v3',credentials=credentials,cache_discovery=False)
    validate_destination(service)
    return service


def qtext(value):return value.replace('\\','\\\\').replace("'","\\'")


def matches(service,parent,name):
    query=f"'{qtext(parent)}' in parents and name = '{qtext(name)}' and trashed = false"
    rows=[];token=None
    while True:
        reply=service.files().list(q=query,spaces='drive',fields='files(id,name,mimeType,size,md5Checksum,appProperties,parents),nextPageToken',pageToken=token,supportsAllDrives=True,includeItemsFromAllDrives=True).execute()
        rows.extend(reply.get('files',[]));token=reply.get('nextPageToken')
        if not token:break
    if len(rows)>1:raise ValueError('Duplicate remote names: '+name)
    return rows


def folder_child(service,parent,name):
    found=matches(service,parent,name)
    if found:
        row=found[0]
        if row['mimeType']!='application/vnd.google-apps.folder' or row.get('appProperties',{}).get('experiment')!=MARKER:
            raise ValueError('Existing remote folder is not owned by this uploader: '+name)
        return row['id']
    return service.files().create(body=dict(name=name,mimeType='application/vnd.google-apps.folder',parents=[parent],appProperties={'experiment':MARKER}),fields='id',supportsAllDrives=True).execute()['id']


def compact_files(source):
    source=Path(source).resolve()
    files=[]
    for p in sorted(source.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        relative=p.relative_to(source)
        if any(part.startswith('.') for part in relative.parts):continue
        if p.suffix.lower() not in ALLOWED:continue
        if p.stat().st_size>64*1024*1024:raise ValueError('Refusing unexpectedly large result: '+str(relative))
        files.append((p,relative))
    if not files:raise ValueError('No compact result files')
    return files


def upload(service,source,experiment):
    # Identity + destination are checked before any creation/update, even when
    # a caller supplies an already-created service object.
    validate_destination(service)
    if experiment not in EXPERIMENTS:raise ValueError('Unknown experiment destination')
    files=compact_files(source)
    from googleapiclient.http import MediaFileUpload
    parent=folder_child(service,FOLDER,EXPERIMENTS[experiment]);folders={():parent};receipts=[]
    for path,relative in files:
        parts=relative.parts[:-1]
        for depth in range(1,len(parts)+1):
            key=parts[:depth]
            if key not in folders:folders[key]=folder_child(service,folders[key[:-1]],key[-1])
        target=folders[parts];found=matches(service,target,path.name);md5=digest_file(path);size=path.stat().st_size
        if found:
            row=found[0]
            if row.get('appProperties',{}).get('experiment')!=MARKER:raise ValueError('Refusing to overwrite unrelated remote file')
            file_id=row['id']
        else:
            # Reserve a stable ID before media transfer. Subsequent invocations
            # update this same file, including an interrupted empty placeholder.
            row=service.files().create(body=dict(name=path.name,parents=[target],appProperties={'experiment':MARKER}),fields='id',supportsAllDrives=True).execute()
            file_id=row['id']
        if not found or row.get('md5Checksum')!=md5 or int(row.get('size',-1))!=size:
            request=service.files().update(fileId=file_id,media_body=MediaFileUpload(str(path),resumable=True,chunksize=8*1024*1024),fields='id,size,md5Checksum,parents',supportsAllDrives=True)
            done=None
            while done is None:
                status,done=request.next_chunk(num_retries=4)
                if status:print(f'UPLOAD {relative}: {status.progress():.0%}',flush=True)
        check=service.files().get(fileId=file_id,fields='id,size,md5Checksum,parents',supportsAllDrives=True).execute()
        if check['id']!=file_id or int(check['size'])!=size or check.get('md5Checksum')!=md5 or target not in check['parents']:
            raise ValueError('Uploaded file ID, size, hash or parent verification failed: '+str(relative))
        receipt=dict(path=str(relative),file_id=file_id,bytes=size,md5=md5,parent=target);receipts.append(receipt)
        print('VERIFIED UPLOAD',json.dumps(receipt),flush=True)
    return dict(account=EMAIL,destination=FOLDER,experiment_folder=parent,files=receipts)


def manual_zip(source,path):
    path=Path(path)
    if path.exists():raise ValueError('ZIP exists; choose a new filename')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for p,relative in compact_files(source):archive.write(p,str(relative))
    print(f'LOCAL ZIP ONLY (not uploaded): {path}, {path.stat().st_size} bytes. Download it and upload manually while signed in as {EMAIL}.',flush=True)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path);p.add_argument('--experiment',choices=EXPERIMENTS)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--manual-zip',type=Path)
    a=p.parse_args()
    if a.manual_zip:
        if not a.source:p.error('--source required')
        manual_zip(a.source,a.manual_zip);return
    try:
        service=connect()
        if a.verify_only:return
        if not a.source or not a.experiment:p.error('--source and --experiment required')
        result=upload(service,a.source,a.experiment)
        print('ALL FILES API-VERIFIED:',json.dumps(result,indent=2),flush=True)
    except Exception as exc:
        print(f'UPLOAD FAILED; local results are preserved. No fallback account used. {exc}\nUse --source PATH --manual-zip /content/experiment-results.zip and google.colab.files.download for manual School-account upload.',flush=True)
        raise


if __name__=='__main__':main()
