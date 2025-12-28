"""
CID-DIMM训练监控工具
用于在训练过程中监控CID-DIMM模块的关键指标
"""

import torch

class CIDMonitor:
    """监控CID-DIMM训练状态"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.alpha_values = []
        self.m_v_means = []
        self.loss_ratio_values = []
        self.loss_itm_values = []
        self.loss_fuse_values = []
    
    def update(self, model, loss_fuse, loss_ratio, loss_itm, m_v=None):
        """
        更新监控数据
        
        Args:
            model: RCLMuFN模型
            loss_fuse: 分类损失
            loss_ratio: CID比例损失
            loss_itm: CID对比损失
            m_v: 一致性mask (可选)
        """
        # 记录alpha
        alpha = model.alpha.item()
        self.alpha_values.append(alpha)
        
        # 记录各种loss
        self.loss_fuse_values.append(loss_fuse.item() if torch.is_tensor(loss_fuse) else loss_fuse)
        self.loss_ratio_values.append(loss_ratio.item() if torch.is_tensor(loss_ratio) else loss_ratio)
        self.loss_itm_values.append(loss_itm.item() if torch.is_tensor(loss_itm) else loss_itm)
        
        # 记录m_v均值（如果提供）
        if m_v is not None:
            self.m_v_means.append(m_v.mean().item())
    
    def get_summary(self):
        """获取统计摘要"""
        if len(self.alpha_values) == 0:
            return "No data collected"
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              CID-DIMM 训练监控报告                           ║
╠══════════════════════════════════════════════════════════════╣
║ Alpha (学习到的融合系数):                                     ║
║   当前值: {self.alpha_values[-1]:.6f}                          ║
║   变化范围: [{min(self.alpha_values):.6f}, {max(self.alpha_values):.6f}]      ║
║                                                              ║
║ Loss_fuse (分类损失):                                         ║
║   平均值: {sum(self.loss_fuse_values)/len(self.loss_fuse_values):.4f}          ║
║   最新值: {self.loss_fuse_values[-1]:.4f}                     ║
║                                                              ║
║ Loss_ratio (一致性比例损失):                                  ║
║   平均值: {sum(self.loss_ratio_values)/len(self.loss_ratio_values):.4f}        ║
║   最新值: {self.loss_ratio_values[-1]:.4f}                    ║
║                                                              ║
║ Loss_itm (对比学习损失):                                      ║
║   平均值: {sum(self.loss_itm_values)/len(self.loss_itm_values):.4f}            ║
║   最新值: {self.loss_itm_values[-1]:.4f}                      ║
"""
        
        if len(self.m_v_means) > 0:
            summary += f"""║                                                              ║
║ M_v均值 (平均一致性):                                         ║
║   平均值: {sum(self.m_v_means)/len(self.m_v_means):.4f} (目标: 0.3)              ║
║   最新值: {self.m_v_means[-1]:.4f}                            ║
"""
        
        summary += """╚══════════════════════════════════════════════════════════════╝
"""
        
        return summary
    
    def check_health(self):
        """检查训练健康状态"""
        if len(self.alpha_values) == 0:
            return "⚠️  无数据"
        
        issues = []
        
        # 检查alpha是否在合理范围
        alpha = self.alpha_values[-1]
        if alpha < -0.1:
            issues.append(f"❌ Alpha为负值({alpha:.4f})，CID信号反向作用")
        elif alpha > 1.0:
            issues.append(f"⚠️  Alpha过大({alpha:.4f})，可能过度依赖CID")
        elif abs(alpha) < 0.001 and len(self.alpha_values) > 100:
            issues.append(f"⚠️  Alpha接近0({alpha:.6f})，CID未学到有用信号")
        else:
            issues.append(f"✓ Alpha正常({alpha:.4f})")
        
        # 检查loss_itm
        if len(self.loss_itm_values) > 0:
            loss_itm = self.loss_itm_values[-1]
            if loss_itm < 0.01:
                issues.append(f"⚠️  loss_itm过小({loss_itm:.4f})，正负样本区分不明显")
            elif loss_itm > 0.5:
                issues.append(f"❌ loss_itm过大({loss_itm:.4f})，mask可能崩溃")
            else:
                issues.append(f"✓ loss_itm正常({loss_itm:.4f})")
        
        # 检查m_v均值
        if len(self.m_v_means) > 10:
            m_v_std = torch.tensor(self.m_v_means[-10:]).std().item()
            if m_v_std < 0.001:
                issues.append(f"⚠️  m_v方差过小({m_v_std:.6f})，可能卡在局部最优")
            else:
                issues.append(f"✓ m_v变化正常(std={m_v_std:.4f})")
        
        return "\n".join(issues)

# 全局监控器实例
cid_monitor = CIDMonitor()
