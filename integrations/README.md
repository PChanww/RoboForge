# 用 SmolVLA 控制仿真

> **历史适配记录（2026-11-09）**：本页的 z+0.60 和七维到四维映射只对应旧龙门原型及该检查点。接入当前项目时，读取所选机器人、相机和控制器的绑定，并按[模型接入指南](../docs/MODEL_INTEGRATION.zh-CN.md)实现适配。

控制器运行在独立 Python 进程中。它从环境获取前视 RGB、腕部 RGB、夹爪状态与英文任务指令，每次输出 50 个七维动作。默认执行前 5 个动作，再读新的观测。每次请求携带当前 episode 和 tick，若有人重置或抢先执行了动作，本轮会中止。

本机已运行标准杯、种子 0 的完整 180 步。RTX 4070 Laptop 上共调用模型 36 次，首次推理 3.221 秒，其余 35 次的中位数为 0.570 秒。末端和夹爪执行了动作，杯子没有被接触或夹起。将相同动作送至第二个独立仿真进程后，最终观测和评分完全一致。详细动作及结果见 [rollout-seed0.json](rollout-seed0.json)。

在准备好的这台电脑上，启动仿真服务后，从工作区根目录运行以下 PowerShell 命令。首次导入 WSL 中的依赖及加载模型可能需要几分钟。运行命令会重置当前场景。

```powershell
wsl -d Ubuntu-22.04 -- bash outputs/vla-lab/integrations/run-local.sh --asset standard --seed 0 --steps 180 --report work/policy/my-rollout.json
```

如已把终端打开在项目的 `vla-lab` 目录中，将脚本路径改成 `integrations/run-local.sh`。脚本会定位该工作区的模型文件。模型约 907 MB，保存在工作区的 `work/smolvla-libero`，未复制进项目源码。

## 在其他机器上运行

安装 Python 3.11、兼容的 PyTorch/CUDA，并在独立 Python 环境中安装 `lerobot[smolvla]==0.4.4` 及 `transformers==4.57.3`。随后从 Hugging Face 获取两个目录。第二个目录只需要分词器、处理器及配置文件。

```bash
hf download lerobot/smolvla_libero --local-dir weights/smolvla-libero
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --include '*.json' '*.txt' --local-dir weights/smolvlm-processor
python integrations/smolvla.py --weights weights/smolvla-libero --processor weights/smolvlm-processor --url http://127.0.0.1:8765 --seed 0 --steps 180 --report rollout.json
```

也可从 [ModelScope 策略权重](https://modelscope.cn/models/lerobot/smolvla_libero) 和 [处理器](https://modelscope.cn/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) 下载同名文件。本机验证时 `model.safetensors` 的 SHA-256 为 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。客户端严格装载全量权重并使用 checkpoint 自带的均值和标准差。

## 为什么动作执行了，杯子仍可能没被拿到

这个 checkpoint 训练在 LIBERO 的 Franka Panda 上。当前桌面用三轴龙门夹爪，因此在输入端将桌面坐标的 z 加 0.60 米、夹持方向近似为朝下、左右手指投影到 ±0.04 米。在输出端保留平移动作、反转夹爪开合符号并裁剪到环境范围；三维旋转无法在该机构执行。两台相机的位置、末端坐标和实际运动步长也与训练分布不同。报告中的 `raw7` 保存模型原始输出，`applied4` 保存实际发给仿真的指令，可以据此重放。只有仿真返回 `metrics.success=true` 才算放置成功。这一局能验证外部 VLA 的传感与动作链路，单局结果不足以评估泛化成功率。

参考 [LeRobot 模型卡](https://huggingface.co/lerobot/smolvla_libero) 和 [LeRobot 0.4.4 的 LIBERO 状态转换](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/processor/env_processor.py)。
