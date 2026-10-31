"""常驻推理引擎的 OpenAI 兼容 HTTP 适配器。

Module: ``coffin.entrypoints.openai``。把 OpenAI 报文翻译成引擎 socket 上
的请求，是本仓库唯一接触 HTTP 的地方。

Constraints:
    * 线格式按 OpenAI 报文来：字段名 ``tool_calls`` / ``finish_reason`` 等
      一律不改写，客户端依赖它们。
    * 这里的任何异常都必须转成 HTTP 响应，进程不能因为一个坏请求退出。
"""
