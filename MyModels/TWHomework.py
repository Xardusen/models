class ListNode(object):
    """链表节点数据类型，用于connections输入格式检验"""
    def __init__(self, value=None):
        self.val = value
        self.next = None

    @staticmethod
    def default_circle():
        """创建一个四节点循环链表"""
        p1 = ListNode(',')
        p2 = ListNode(' ')
        p3 = ListNode(',')
        p4 = ListNode(';')
        p1.next = p2
        p2.next = p3
        p3.next = p4
        p4.next = p1
        return p1


class Maze(object):
    """
    创建Maze类型
    maze：用于保存“迷宫”的(2m + 1) x (2n + 1)矩阵，其中“墙”值为1，“通道”值为0
    connection: 用于存储迷宫连通节点信息的矩阵，其每个元素为1 x 4的链表：[x, y, u, v]对应于
        节点(x, y)与节点(u, v)相连通
    size：用于保存迷宫输入大小信息的1 x 2矩阵：[m, n]
    """
    def __init__(self):
        self.maze = []
        self.connections = []
        self.size = [0, 0]

    def reset(self):
        """对已经创建的Maze进行重置（去除所有通道）"""
        self.maze = []
        if self.size:
            for i in range(2 * self.size[0] + 1):
                row_temp = []
                for j in range(2 * self.size[1] + 1):
                    row_temp.append(0 if i % 2 == 1 and j % 2 == 1 else 1)
                self.maze.append(row_temp)

    def create(self, input_string):
        """根据input_string将maze初始化为一个(2m + 1) x (2n + 1)的矩阵"""
        if len(input_string.split(' ')) != 2:
            print("Incorrect command format.")  # 输入中没有空格字符报错
        else:
            col, row = input_string.split(' ')
            try:
                size = [int(col), int(row)]
            except ValueError:
                print("Invalid number format.")  # 字符串无法数字化报错
            else:
                if size[0] <= 0 or size[1] <= 0 or size[0] > 100 or size[1] > 100:
                    print("Number out of range.")  # 数字小于0或者大于阈值报错（此处设为100）
                else:
                    self.size = size
                    self.reset()

    def add_connections(self, input_connections):
        """根据input_connections在已经初始化的maze中加入通道"""
        connections = []
        if not input_connections:
            pass
        elif not input_connections == input_connections.strip():
            print("Incorrect command format.")  # 输入两端有空格报错
        else:
            flag = ListNode.default_circle()
            for char in input_connections:
                if char == flag.val:
                    flag = flag.next
            if flag.val != ';':
                print("Incorrect command format.")  # 输入格式错误报错
            else:
                pairs = input_connections.split(';')
                for pair in pairs:
                    p1, p2 = pair.split(' ', 1)
                    connections.append(p1.split(',', 1) + p2.split(',', 1))
                try:
                    for i in range(len(connections)):
                        for j in range(len(connections[0])):
                            connections[i][j] = int(connections[i][j])
                except ValueError:
                    print("Invalid number format.")  # 坐标无法数字化报错
                else:
                    self.connections = connections
                    for link in self.connections:
                        if min(link) < 0 or max(link[0], link[2]) > self.size[0] - 1 or max(link[1], link[3]) > self.size[1] - 1:
                            print("Number out of range.")  # 数字超出范围报错，即点坐标应该在m x n矩阵内
                        elif (link[0] - link[2]) ** 2 + (link[1] - link[3])** 2 == 1:
                            if link[0] == link[2]:
                                self.maze[2 * link[0] + 1][link[1] + link[3] + 1] = 0
                            else:
                                self.maze[link[0] + link[2] + 1][2 * link[1] + 1] = 0
                        else:
                            print("Maze format error.")  # 两点非相邻报错

    def render(self):
        """将maze渲染为由[W],[R]组成的二维图像，返回值类型为字符串"""
        render_string = ""
        for line in self.maze:
            for order in range(len(line)):
                render_string += "[W]" if line[order] == 1 else "[R]"
                render_string += '\n' if order == len(line) - 1 else " "
        return render_string


def main():
    maze = Maze()
    while not maze.maze:
        maze.create(input("Enter size : "))
    while not maze.connections:
        maze.add_connections(input("Enter connections : "))
    print(maze.render())
if __name__ == "__main__":
    main()
