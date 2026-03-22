import torch
def corners_to_heatmap(corners_list,height,width,sigma=2.0,num_classes=8):
    heatmap = torch.zeros((num_classes,height,width),dtype=torch.float32)
    for class_id in range(num_classes):
        corners = corners_list[class_id]
        for corner in corners:
            x,y = corner
            x = torch.clip(x,0,width-1).long()
            y = torch.clip(y,0,height-1).long()
            xx,yy = torch.meshgrid(torch.arange(width),torch.arange(height),indexing='xy')
            gaussian = torch.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2))
            heatmap[class_id] = torch.maximum(heatmap[class_id],gaussian)
    return heatmap


                