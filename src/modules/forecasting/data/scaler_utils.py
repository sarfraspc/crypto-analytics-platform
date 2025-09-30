from pathlib import Path
import joblib
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, List




def _scaler_path_for(base_dir: Path, symbol: Optional[str], global_name: str = 'scaler_global.pkl'):
    base_dir = Path(base_dir)
    if symbol:
        safe = symbol.replace('/', '_').upper()
        return base_dir / f'scaler_{safe}.pkl'
    return base_dir / global_name




def save_scaler_with_meta(path: Path, scaler: MinMaxScaler, cols_order: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    meta = {'cols_order': cols_order, 'scaler_type': type(scaler).__name__, 'saved_at': pd.Timestamp.utcnow().isoformat()}
    path.with_suffix('.json').write_text(json.dumps(meta))




def load_scaler_with_meta(path: Path) -> Tuple[Optional[MinMaxScaler], Optional[List[str]]]:
    meta_p = path.with_suffix('.json')
    if not path.exists() or not meta_p.exists():
        return None, None
    scaler = joblib.load(path)
    meta = json.loads(meta_p.read_text())
    return scaler, meta.get('cols_order')
