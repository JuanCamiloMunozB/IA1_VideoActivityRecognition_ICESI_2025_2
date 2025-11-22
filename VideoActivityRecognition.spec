# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Users\\Lenovo\\final\\IA1_VideoActivityRecognition_ICESI_2025_2\\Entrega3\\experiments\\models', 'Entrega3/experiments/models'), ('C:\\Users\\Lenovo\\final\\IA1_VideoActivityRecognition_ICESI_2025_2\\Entrega3\\experiments\\results', 'Entrega3/experiments/results'), ('C:\\Users\\Lenovo\\final\\IA1_VideoActivityRecognition_ICESI_2025_2\\Entrega2\\experiments\\results', 'Entrega2/experiments/results'), ('C:\\Users\\Lenovo\\final\\IA1_VideoActivityRecognition_ICESI_2025_2\\Entrega2\\src', 'Entrega2/src'), ('C:\\Users\\Lenovo\\final\\IA1_VideoActivityRecognition_ICESI_2025_2\\Entrega3\\src', 'Entrega3/src')]
binaries = []
hiddenimports = ['sklearn.utils._typedefs', 'sklearn.utils._cython_blas', 'sklearn.neighbors._partition_nodes', 'sklearn.tree._utils', 'sklearn.utils._weight_vector', 'sklearn.pipeline', 'sklearn.preprocessing', 'sklearn.preprocessing._data', 'sklearn.preprocessing._encoders', 'sklearn.preprocessing._label', 'sklearn.feature_selection', 'sklearn.feature_selection._univariate_selection', 'sklearn.svm', 'sklearn.svm._classes', 'sklearn.svm._libsvm', 'sklearn.svm._liblinear', 'sklearn.metrics', 'sklearn.metrics._classification', 'cv2', 'cv2.data', 'mediapipe', 'mediapipe.python', 'numpy', 'numpy.core._methods', 'numpy.lib.format', 'pandas', 'pandas._libs.tslibs.timedeltas', 'joblib', 'xgboost', 'scipy', 'scipy.special.cython_special', 'scipy.sparse.csgraph._validation']
datas += collect_data_files('cv2')
binaries += collect_dynamic_libs('cv2')
tmp_ret = collect_all('mediapipe')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\Users\\Lenovo\\final\\IA1_VideoActivityRecognition_ICESI_2025_2\\app_entry.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VideoActivityRecognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
