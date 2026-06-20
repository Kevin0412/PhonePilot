"""PhonePilot —— 实验性 Android 手机 agent 框架。

包根保持轻量：不在此 import 任何平台/重型依赖（cv2、onnxruntime、http 适配器），
以便纯逻辑核心（types/signature/local_rules/safety/risk/budget）可零依赖 import，
也便于后续 Chaquopy 嵌入 / Kotlin 移植时隔离平台运行时。
"""

__version__ = "0.1.0"
