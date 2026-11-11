import argparse
from .preflight import check
from .storage import Store
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo');p.add_argument('--run');a=p.parse_args()
    check(a.root,Store(a.root,a.repo,a.run))
