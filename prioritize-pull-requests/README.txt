organization爬取日期为2017年12月20日
organization里面存在很多无效组织。比如https://api.github.com/orgs/nmh48cmdv
12月24日15点开始爬取pull request的数量


根据460万项目的pull request的数目来看：详见(analysis/pull_request_number.txt文件)
1. 400多万的项目没有pull request记录，即pull request的数目为0，也就是说这样的项目本身只利用了github的群体管理功能，而并没有按照pull-based的开发模式的流程进行开发。
2. 峰值为54941个。有36个项目的pull request的数目过万。
3. 目前预测优先级和预测accept的概率的两个数据集构造应选取不同的项目，其中预测优先级的数据对project的pull request的要求比较高。


下一步计划：写两个通用的数据集构造脚本，阈值为pull request数目。

