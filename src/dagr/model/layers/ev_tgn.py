import torch

from torch_geometric.data import Batch, Data
from dagr.graph.ev_graph import SlidingWindowGraph


def _get_value_as_int(obj, key):
    val = getattr(obj, key)
    return val if type(val) is int else val[0]

def denormalize_pos(events):
    if hasattr(events, "pos_denorm"):
        return events.pos_denorm

    denorm = torch.tensor([int(events.width[0]), int(events.height[0]), int(events.time_window[0])], device=events.pos.device)
    return (denorm.view(1,-1) * events.pos + 1e-3).int()

#将事件数据转换为图数据，每个事件变成一个节点，时间和空间相近的事件之间建立有向边
class EV_TGN(torch.nn.Module): 
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.radius = args.radius
        self.max_neighbors = args.max_neighbors
        self.max_queue_size = 128
        self.graph_creators = None
    #初始化图创建器，创建一个滑动窗口图创建器，用于生成图结构。该图创建器使用事件数据的宽度、高度、最大邻居数、最大队列大小、半径和时间窗口等参数进行初始化。
    def init_graph_creator(self, data):
        delta_t_us = int(self.radius * _get_value_as_int(data, "time_window"))
        radius = int(self.radius * _get_value_as_int(data, "width")+1)
        batch_size = data.num_graphs
        width = int(_get_value_as_int(data, "width"))
        height = int(_get_value_as_int(data, "height"))
        self.graph_creators = SlidingWindowGraph(width=width, height=height,
                                                 max_num_neighbors=self.max_neighbors,
                                                 max_queue_size=self.max_queue_size,
                                                 batch_size=batch_size,
                                                 radius=radius, delta_t_us=delta_t_us)

    def forward(self, events: Data, reset=True):
        if events.batch is None:
            events = Batch.from_data_list([events])

        # before we start, are the new events used to generate the graph, or are the new nodes attached to the network?
        # if the first, then don't delete old events, if the second, delete as many events as are coming in.
        if self.graph_creators is None:
            self.init_graph_creator(events) #初始化图创建器，创建一个滑动窗口图创建器，用于生成图结构。该图创建器使用事件数据的宽度、高度、最大邻居数、最大队列大小、半径和时间窗口等参数进行初始化。
        else:
            if reset: #reset是一个布尔值，表示是否重置图创建器
                self.graph_creators.reset()
        #pos是事件数据的位置信息，包含事件的空间位置和时间信息。通过将事件数据的宽度、高度和时间窗口等参数进行归一化处理，得到事件在图中的位置。
        pos = denormalize_pos(events)
        #pos = torch.cat([events.batch.view(-1,1), pos, events.x.int()], dim=1).int()
        # properties of the edges
        # src_i <= dst_i
        # dst_i <= dst_j if i<j
        events.edge_index = self.graph_creators.forward(events.batch.int(), pos, delete_nodes=False, collect_edges=reset)
        events.edge_index = events.edge_index.long()

        return events # 最终的event被转化为图数据（节点），图数据对象包含了事件数据的节点特征、边索引、边属性等信息，可以用于后续的图神经网络模型进行训练和推理。