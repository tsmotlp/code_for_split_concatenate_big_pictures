import torch
def chop_forward(x, model, scale=4, rows=3, cols=4, h_shave=19, w_shave=4):
    b, c, h, w = x.size()
    print('b, c, h, w=', b, c, h, w)
    h_split, w_split = h // rows, w // cols    # 平均分成几行几列
    # h_size, w_size = h_split + h_shave, w_split + w_shave
    
    inputlist = []
    for i in range(rows):
        for j in range(cols):
            print('i, j:', i, j)
            if i==0: # 第一行
                if j == 0: # 第一列
                    inputlist.append(x[:, :, 0:h_split + 2 * h_shave, 0:w_split + 2 * w_shave])
                elif j == cols-1: # 最后一列
                    inputlist.append(x[:, :, 0:h_split + 2 * h_shave, (w - w_split - 2 * w_shave):w])
                else:  # 其他列
                    inputlist.append(x[:, :, 0:h_split + 2 * h_shave, (j * w_split - w_shave):((j+1) * w_split + w_shave)])
            elif i == rows-1: # 最后一行
                if j == 0:  # 第一列
                    inputlist.append(x[:, :, (h - h_split - 2 * h_shave):h, 0:w_split + 2 * w_shave])
                elif j == cols-1:  # 最后一列
                    inputlist.append(x[:, :, (h - h_split - 2 * h_shave):h, (w - w_split - 2 * w_shave):w])
                else:  # 其他列
                    inputlist.append(x[:, :, (h - h_split - 2 * h_shave):h, (j * w_split - w_shave):((j+1) * w_split + w_shave)])
            else:  # 其他行
                if j == 0:  # 第一列
                    inputlist.append(x[:, :, (i * h_split - h_shave):((i+1) * h_split + h_shave), 0:w_split + 2 * w_shave])
                elif j == cols-1:  # 最后一列
                    inputlist.append(x[:, :, (i * h_split - h_shave):((i+1) * h_split + h_shave), (w - w_split - 2 * w_shave):w])
                else:
                    inputlist.append(x[:, :, (i * h_split - h_shave):((i+1) * h_split + h_shave), (j * w_split - w_shave):((j+1) * w_split + w_shave)])
            
    print("inputlist[0] shape:", inputlist[0].shape)
    outputlist = []
    for s in range(0, rows * cols):
        with torch.no_grad():
            input_batch = inputlist[s]#torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            output_batch = model(input_batch)
        outputlist.append(output_batch)
    # else:
    #     outputlist = [
    #         chop_forward(patch, model, scale, shave, min_size, nGPUs) \
    #         for patch in inputlist]
    
    h, w = scale * h, scale * w
    h_split, w_split = scale * h_split, scale * w_split
    # h_size, w_size = scale * h_size, scale * w_size
    h_shave *= scale
    w_shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))
    for k in range(rows):
        for v in range(cols):
            print('k, v:', k, v)
            if k == 0:    # 第一行
                if v == 0:    # 第一列
                    output[:, :, 0:h_split, v*w_split:(v+1)*w_split] = outputlist[k*cols+v][:, :, 0:h_split, 0:w_split]
                elif v == cols-1:    # 最后一列
                    output[:, :, k * h_split:(k+1)*h_split, v*w_split:(v+1)*w_split] = outputlist[k*cols+v][:, :, 0:h_split, 2 * w_shave:w_split + 2 * w_shave]
                else:    # 其他列
                    output[:, :, k * h_split:(k+1)*h_split, v*w_split:(v+1)*w_split] = outputlist[k*cols+v][:, :, 0:h_split, w_shave:w_split + w_shave]
            elif k == rows-1:    # 最后一行
                if v == 0:    # 第一列
                    output[:, :, k * h_split:(k+1)*h_split, v*w_split:(v+1)*w_split] = outputlist[k*cols+v][:, :, 2 * h_shave:h_split + 2 * h_shave, 0:w_split]
                elif v == cols-1:    # 最后一列
                    output[:, :, k * h_split:(k+1)*h_split, v*w_split:(v+1)*w_split] = outputlist[k*cols+v][:, :, 2 * h_shave:h_split + 2 * h_shave, 2 * w_shave:w_split + 2 * w_shave]
                else:    # 其他列
                    output[:, :, k * h_split:(k+1)*h_split, v*w_split:(v+1)*w_split] = outputlist[k*cols+v][:, :, 2 * h_shave:h_split + 2 * h_shave, w_shave:w_split + w_shave]
            else:    # 其他行
                if v == 0:    # 第一列
                    output[:, :, k * h_split:(k+1)*h_split, v*w_split:(v+1)*w_split] = outputlist[k*cols+v][:, :, h_shave:h_split + h_shave, 0:w_split]
                elif v == cols-1:    # 最后一列
                    output[:, :, k * h_split:(k+1)*h_split, v*w_split:(v+1)*w_split] = outputlist[k*cols+v][:, :, h_shave:h_split + h_shave, 2 * w_shave:w_split + 2 * w_shave]
                else:    # 其他列
                    output[:, :, k * h_split:(k+1)*h_split, v*w_split:(v+1)*w_split] = outputlist[k*cols+v][:, :, h_shave:h_split + h_shave, w_shave:w_split + w_shave]        
    return output