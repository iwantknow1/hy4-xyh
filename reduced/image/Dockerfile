FROM vllm-ascend-hy4:a5-cann9.1
ARG SOURCE_MANIFEST_SHA256
LABEL org.opencontainers.image.title="HY4 A5 reduced-model verification environment" \
      hy4.validation.scope="functional_smoke" \
      hy4.source.manifest.sha256="${SOURCE_MANIFEST_SHA256}"
COPY src/vllm/ /vllm-workspace/vllm/
COPY src/vllm-ascend/ /vllm-workspace/vllm-ascend/
COPY .venv/ /codex/.venv/
COPY image-deps/triton/ /usr/local/python3.12.13/lib/python3.12/site-packages/triton/
COPY image-deps/triton_ascend-3.6.0.dist-info/ /usr/local/python3.12.13/lib/python3.12/site-packages/triton_ascend-3.6.0.dist-info/
COPY tools/uv /codex/tools/uv
COPY tooling/skills/ /codex/tooling/skills/
COPY scripts/ /codex/scripts/
COPY model_openbox/ /codex/model_openbox/
COPY logs/resume-exclusive/ /codex/logs/resume-exclusive/
COPY models/Hy4-2L-MTP/ /codex/models/Hy4-2L-MTP/
COPY image/entrypoint.sh /usr/local/bin/hy4-entrypoint
COPY image/serve.sh /usr/local/bin/hy4-serve
COPY image/README.zh-CN.md /codex/IMAGE-README.zh-CN.md
COPY image/source-manifest.json /codex/source-manifest.json
COPY performance/ /codex/performance/
RUN chmod 755 /usr/local/bin/hy4-entrypoint /usr/local/bin/hy4-serve
ENV PATH="/codex/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
WORKDIR /codex
ENTRYPOINT ["/usr/local/bin/hy4-entrypoint"]
CMD ["bash"]
