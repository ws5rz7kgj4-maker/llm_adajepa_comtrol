# GitHub upload / 上传说明

Suggested repository name / 建议仓库名：`llm-adajepa-control`

Suggested description / 建议简介：

> VLM-guided adaptive robot control: asynchronous planning, reproducible failure analysis, and bilingual interview documentation.

## 中文

1. 解压 ZIP，进入 `llm-adajepa-control` 文件夹。
2. 在 GitHub 创建新仓库。上传时应让 `README.md` 位于仓库根目录，不要只把 ZIP 文件放上去。
3. 使用 Git 推送可完整保留 `.github`、`.gitignore` 等配置；网页上传时也要确保这些文件在内。
4. 打开仓库主页检查中英文链接、结果图和公式。CI 配置已提供，远端是否通过以 GitHub Actions 的实际运行结果为准。

从解压后的仓库根目录执行：

```bash
git init -b main
git add .
git commit -m "Package bilingual LLM and adaptive-control research portfolio"
git remote add origin https://github.com/YOUR_USERNAME/llm-adajepa-control.git
git push -u origin main
```

将 `YOUR_USERNAME` 换成自己的 GitHub 用户名。仓库应先在 GitHub 建好，避免额外初始化 README 导致分支冲突。若已有 Git 仓库，按现有远端与分支操作，不重复添加 origin。

面试分享仓库主页即可；优先让对方阅读 README、运行离线演示，再查看研究代码。本包已经准备好上传，但整理过程没有自动发布远程仓库。

## English

Extract the ZIP and make the contents of `llm-adajepa-control` the repository root, with `README.md` at the top level. Create an empty GitHub repository, replace `YOUR_USERNAME` in the commands above, then commit and push.

Git preserves the included workflow and ignore file. If using browser upload, include them as well. Check the README, bilingual links and result figure after publication. The workflow is supplied, but only an actual GitHub Actions run establishes remote CI status.

This package has not itself been published to a remote repository. It is ready for review and upload. For an existing repository, use its existing branch and remote configuration.
