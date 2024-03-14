import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import FeatureTransformer3D, FeatureTransformer3D_PT
from models.matching import global_correlation_softmax_3d, SelfCorrelationSoftmax3D
from models.backbone import DGCNN#, PointNet, MLP

class DiffSF(nn.Module):
    def __init__(self,
                 backbone='DGCNN',
                 channels=128,
                 ffn_dim_expansion=4,
                 num_transformer_pt_layers=1,
                 num_transformer_layers=8,
                 self_condition = True,
                 ):
        super(DiffSF, self).__init__()

        self.backbone = backbone
        self.channels = channels
        self.self_condition = self_condition

        # PointNet
        if self.backbone=='PointNet':
            self.pointnet0 = PointNet(output_channels = self.channels)
            self.pointnet1 = PointNet(output_channels = self.channels)
            channels = self.channels
        #MLP
        if self.backbone=='MLP':
            self.mlp0 = MLP(output_channels = self.channels)
            self.mlp1 = MLP(output_channels = self.channels)
            channels = self.channels
        # DGCNN
        if self.backbone=='DGCNN':
            self.DGCNN0 = DGCNN(output_channels = self.channels, k=16)
            self.DGCNN1 = DGCNN(output_channels = self.channels, k=16)
            channels = self.channels

        self.num_transformer_layers = num_transformer_layers
        self.num_transformer_pt_layers = num_transformer_pt_layers

        # Transformer
        if self.num_transformer_layers > 0:
            self.transformer1 = FeatureTransformer3D(num_layers=num_transformer_layers,
                                   d_model=channels,
                                    ffn_dim_expansion=ffn_dim_expansion,
                                    )
        if self.num_transformer_pt_layers > 0:
            self.transformer_PT = FeatureTransformer3D_PT(num_layers=num_transformer_pt_layers,
                                                        d_points=channels,
                                                        ffn_dim_expansion=ffn_dim_expansion,
                                                        k=16,
                                                        )
        # self correlation with self-feature similarity
        self.feature_flow_attn0 = SelfCorrelationSoftmax3D(in_channels=self.channels)
        self.feature_flow_attn1 = SelfCorrelationSoftmax3D(in_channels=self.channels)



    def forward(self, pred, time, pcs, 
                **kwargs,
                ):
        
        flow_preds = []
        xyzs1, xyzs2 = pcs[:,0:3,:], pcs[:,3:6,:]

        if self.backbone=='DGCNN':
            feature1 = self.DGCNN0(xyzs1 + pred)
            feature2 = self.DGCNN0(xyzs2)
        if self.backbone=='PointNet':
            feature1 = self.pointnet0(xyzs1 + pred)
            feature2 = self.pointnet0(xyzs2)
        if self.backbone=='MLP':
            feature1 = self.mlp0(xyzs1 + pred)
            feature2 = self.mlp0(xyzs2)
        flow = global_correlation_softmax_3d(feature1, feature2, xyzs1, xyzs2)[0]
        pred = self.feature_flow_attn0(feature1, flow)

        if self.backbone=='DGCNN':
            feature1 = self.DGCNN1(xyzs1 + pred)
            feature2 = self.DGCNN1(xyzs2)
        if self.backbone=='PointNet':
            feature1 = self.pointnet1(xyzs1 + pred)
            feature2 = self.pointnet1(xyzs2)
        if self.backbone=='MLP':
            feature1 = self.mlp1(xyzs1 + pred)
            feature2 = self.mlp1(xyzs2)
        if self.num_transformer_pt_layers > 0:
            feature1, feature2 = self.transformer_PT(xyzs1, xyzs2, feature1, feature2)
        feature1, feature2 = self.transformer1(feature1, feature2)
        flow = global_correlation_softmax_3d(feature1, feature2, xyzs1, xyzs2)[0]
        pred = self.feature_flow_attn1(feature1, flow)
        flow_preds.append(pred)
        
        return flow_preds