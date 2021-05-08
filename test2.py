import torch




def max_index(list_pro):
    max_indax=torch.argmax(list_pro)
    val, index = torch.topk(list_pro, 2, dim=0, largest=True, sorted=True)
    if list_pro[max_indax]>2.8:
        out=max_indax
        return out.detach().numpy()
    else:
        # XXX=abs(index.detach().numpy()[0] - index.detach().numpy()[1])
        if abs(index.detach().numpy()[0]-index.detach().numpy()[1])==10 or abs(index.detach().numpy()[0]-index.detach().numpy()[1])==9:
            if max_indax>10:
                out=max_indax
                return out.detach().numpy()
            else:
                out=max_indax+abs(index.detach().numpy()[0]-index.detach().numpy()[1])
                return out.detach().numpy()
        else:
            return max_indax.detach().numpy()


if __name__=='__main__':
    a=torch.tensor([1,3.8,3,0.4,0.5,0.6,0.7,0.8,0.8,0.0,3.3,0])
    print(a)
    _,b=torch.topk(a,2,dim=0,largest=True,sorted=True)
    print(abs(b.detach().numpy()[0]-b.detach().numpy()[1])==9)
    print(max_index(a))