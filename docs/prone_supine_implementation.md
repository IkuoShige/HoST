# Pi Robot: うつ伏せ・仰向け両対応 起き上がり実装

## 概要

1つの統合ポリシーで仰向け（supine）・うつ伏せ（prone）両方からの起き上がりを学習させる機能を実装した。

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `legged_gym/legged_gym/envs/pi/pi_config_ground.py` | ランダム初期姿勢の設定を追加 |
| `legged_gym/legged_gym/envs/pi/pi_host_ground.py` | リセット時のランダム姿勢選択ロジックを実装 |
| `legged_gym/legged_gym/utils/isaacgym_utils.py` | `@torch.jit.script` を削除（CUDA互換性修正） |

## 実装詳細

### 1. 設定ファイル (`pi_config_ground.py`)

`init_state` クラスに以下を追加（lines 9-13）:

```python
class init_state( LeggedRobotCfg.init_state ):
    pos = [0.0, 0.0, 0.351]
    rot = [0.0, -1, 0, 1.0]  # デフォルト（仰向け）

    # ランダム初期姿勢の設定
    random_initial_orientation = True
    supine_rot = [0.0, -1, 0, 1.0]  # 仰向け (back down)
    prone_rot = [0.0, 1, 0, 1.0]    # うつ伏せ (face down)
    supine_prone_ratio = 0.5        # 50%仰向け, 50%うつ伏せ
```

### 2. リセット処理 (`pi_host_ground.py`)

`_reset_root_states()` メソッド（lines 514-524）にランダム姿勢選択を追加:

```python
# Randomly select supine/prone orientation
if getattr(self.cfg.init_state, 'random_initial_orientation', False):
    supine_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.init_state.supine_prone_ratio

    supine_quat = torch.tensor(self.cfg.init_state.supine_rot, device=self.device, dtype=torch.float)
    prone_quat = torch.tensor(self.cfg.init_state.prone_rot, device=self.device, dtype=torch.float)

    # Apply supine orientation
    self.root_states[env_ids[supine_mask], 3:7] = supine_quat
    # Apply prone orientation
    self.root_states[env_ids[~supine_mask], 3:7] = prone_quat
```

### 3. JIT互換性修正 (`isaacgym_utils.py`)

`@torch.jit.script` デコレータを削除（line 6）:

```python
# 変更前
@torch.jit.script
def copysign(a, b):
    ...

# 変更後
def copysign(a, b):
    ...
```

## 設定パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `random_initial_orientation` | `True` | ランダム姿勢選択の有効/無効 |
| `supine_rot` | `[0.0, -1, 0, 1.0]` | 仰向けのクォータニオン (x,y,z,w) |
| `prone_rot` | `[0.0, 1, 0, 1.0]` | うつ伏せのクォータニオン (x,y,z,w) |
| `supine_prone_ratio` | `0.5` | 仰向けの割合（0.5 = 50%仰向け, 50%うつ伏せ） |

## 訓練コマンド

```bash
source .venv/bin/activate
python legged_gym/legged_gym/scripts/train.py --task=pi_ground --num_envs=4096 --headless
```

## 検証方法

### 視覚的確認
```bash
python legged_gym/legged_gym/scripts/train.py --task=pi_ground --num_envs=64
```
ロボットが仰向け・うつ伏せ両方から初期化されることを確認。

### 再生テスト
```bash
python legged_gym/legged_gym/scripts/play.py --task=pi_ground --checkpoint_path=<model.pt>
```

## 報酬関数について（変更不要）

既存の報酬関数は姿勢に依存しない設計のため、変更不要:

- `_reward_orientation`: `projected_gravity[:, 2]` を使用 → 直立を報酬
- `_reward_head_height`: 足からの相対高さ → 姿勢非依存
- `_reward_ground_parallel`: 足の平坦さ → 姿勢非依存

curriculum forceの条件 `projected_gravity[:, 2] < -0.8` も両姿勢で正しく動作する。

## 調整のヒント

- **うつ伏せが難しい場合**: `supine_prone_ratio` を下げる（例: 0.3 で30%仰向け, 70%うつ伏せ）
- **訓練時間**: 2姿勢学習のため、イテレーション数を1.5〜2倍に増やすことを検討
- **従来動作に戻す**: `random_initial_orientation = False` に設定

---

## CUDA互換性問題（RTX 5090）

### 問題

RTX 5090 (Blackwell, compute capability 12.0, sm_120) で `torch.jit.script` 使用時にエラー:

```
RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)
```

### 原因

| コンポーネント | バージョン | sm_120対応 |
|--------------|-----------|-----------|
| PyTorchビルド時CUDA | 12.8 | ✅ |
| システムNVRTC | 12.4 | ❌ |

PyTorchは12.8でビルドされているが、JITランタイムコンパイル（NVRTC）はシステムのCUDA 12.4を使用。CUDA 12.4はBlackwell非対応。

### 解決策

**推奨**: Dockerコンテナを CUDA 12.6+ でリビルド

**一時回避策**:
```bash
PYTORCH_JIT=0 python legged_gym/legged_gym/scripts/train.py --task=pi_ground ...
```

---

## ブランチ情報

- ブランチ: `feat/pi-ground-flat-feet`
- 最新コミット: この実装は未コミット状態

---

## 4方向初期姿勢対応 + 足裏接地改善（2025-01更新）

### 課題

1. **足裏接地の問題**: 立ち上がり後、左足がつま先立ちに収束する
2. **初期姿勢の限定**: 仰向け・うつ伏せのみで、左右に倒れた姿勢に対応できない

### 実装内容

#### 1. 4方向初期姿勢の追加

`pi_config_ground.py` の `init_state` クラスを更新:

```python
# Random initial orientation settings (4 directions)
random_initial_orientation = True
supine_rot = [0.0, -1, 0, 1.0]      # 仰向け (back down)
prone_rot = [0.0, 1, 0, 1.0]        # うつ伏せ (face down)
left_side_rot = [1.0, 0, 0, 1.0]    # 左側臥位 (left side down)
right_side_rot = [-1.0, 0, 0, 1.0]  # 右側臥位 (right side down)
orientation_weights = [0.25, 0.25, 0.25, 0.25]  # 各姿勢の確率
```

`pi_host_ground.py` の `_reset_root_states()` を更新:

```python
# Randomly select initial orientation from 4 directions
if getattr(self.cfg.init_state, 'random_initial_orientation', False):
    weights = getattr(self.cfg.init_state, 'orientation_weights', [0.25, 0.25, 0.25, 0.25])

    quats = [
        torch.tensor(self.cfg.init_state.supine_rot, ...),
        torch.tensor(self.cfg.init_state.prone_rot, ...),
        torch.tensor(getattr(self.cfg.init_state, 'left_side_rot', [1.0, 0, 0, 1.0]), ...),
        torch.tensor(getattr(self.cfg.init_state, 'right_side_rot', [-1.0, 0, 0, 1.0]), ...),
    ]

    # Sample orientation index for each env
    orientation_idx = torch.multinomial(
        torch.tensor(weights, device=self.device),
        num_samples=len(env_ids),
        replacement=True
    )

    for i, quat in enumerate(quats):
        mask = orientation_idx == i
        if mask.any():
            self.root_states[env_ids[mask], 3:7] = quat
```

#### 2. 足裏接地報酬の強化

`pi_config_ground.py` の `constraints.scales` を更新:

| パラメータ | 変更前 | 変更後 | 目的 |
|-----------|--------|--------|------|
| `style_ground_parallel` | 8 | 12 | 足が地面と平行になるよう強化 |
| `style_ankle_pitch_neutral` | 2.5 | 5 | 足首ピッチ角を-0.1radに強く誘導 |
| `style_feet_contact_balance` | 2.5 | 5 | 左右の接地力バランスを改善 |

### 新しい設定パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `left_side_rot` | `[1.0, 0, 0, 1.0]` | 左側臥位のクォータニオン (x,y,z,w) |
| `right_side_rot` | `[-1.0, 0, 0, 1.0]` | 右側臥位のクォータニオン (x,y,z,w) |
| `orientation_weights` | `[0.25, 0.25, 0.25, 0.25]` | [仰向け, うつ伏せ, 左側臥, 右側臥]の確率 |

### クォータニオン一覧

| 姿勢 | 説明 | クォータニオン [x, y, z, w] |
|------|------|---------------------------|
| supine | 仰向け（背中が下） | [0.0, -1, 0, 1.0] |
| prone | うつ伏せ（顔が下） | [0.0, 1, 0, 1.0] |
| left_side | 左側臥位（左が下） | [1.0, 0, 0, 1.0] |
| right_side | 右側臥位（右が下） | [-1.0, 0, 0, 1.0] |

### 検証方法

```bash
# 視覚的確認（4方向初期化）
source .venv/bin/activate
python legged_gym/legged_gym/scripts/train.py --task=pi_ground --num_envs=64

# ヘッドレス訓練
python legged_gym/legged_gym/scripts/train.py --task=pi_ground --num_envs=4096 --headless
```

### 調整のヒント

- **特定方向を重視**: `orientation_weights` を調整（例: `[0.4, 0.4, 0.1, 0.1]` で仰向け/うつ伏せ重視）
- **2方向のみに戻す**: `orientation_weights = [0.5, 0.5, 0.0, 0.0]`
- **足裏接地が改善しない場合**: `style_ground_parallel` をさらに上げる（例: 15）

---

## 次のステップ

1. CUDA 12.8でDockerコンテナをリビルド
2. JIT有効状態で訓練実行を確認
3. 4方向からの起き上がり成功率をTensorBoardで確認
4. 足裏接地の改善を視覚的に確認
5. 必要に応じて `orientation_weights` や報酬スケールを調整
