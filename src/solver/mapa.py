import rclpy
from rclpy.node import Node
from cg_interfaces.srv import MoveCmd, GetMap
import numpy as np
import time

def validar_mapa(mapa):
    """Valida a estrutura do mapa."""
    if not all(isinstance(row, list) for row in mapa):
        raise ValueError("Mapa deve ser uma lista de listas.")
    if not all(len(row) == len(mapa[0]) for row in mapa):
        raise ValueError("Todas as linhas do mapa devem ter o mesmo comprimento.")
    return True

class RobotClient(Node):
    def __init__(self):
        super().__init__('robot_client')

        # Cliente para o serviço de movimentação
        self.move_client = self.create_client(MoveCmd, '/move_command')
        while not self.move_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando pelo serviço de movimentação...')
        
        # Cliente para o serviço de mapa
        self.map_client = self.create_client(GetMap, '/get_map')
        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando pelo serviço de mapa...')
        
        self.target_position = [18, 18]
        self.map_data = 'default.csv'
        self.path = []

    def get_map(self):
        """Obtém o mapa e identifica as posições de início e objetivo."""
        request = GetMap.Request()
        future = self.map_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response:
            self.get_logger().info(f"Mapa recebido.")
            try:
                # Converte o mapa para matriz bidimensional
                shape = tuple(response.occupancy_grid_shape)
                flattened_grid = response.occupancy_grid_flattened
                
                # Garante que o mapa seja interpretado como strings
                self.map_data = np.array(flattened_grid, dtype=str).reshape(shape)
                
                # Identificar posições do robô e objetivo
                start = np.argwhere(self.map_data == 'r')
                goal = np.argwhere(self.map_data == 't')
                
                if len(start) > 0 and len(goal) > 0:
                    self.start_position = tuple(start[0])  # Posição inicial
                    self.goal_position = tuple(goal[0])   # Posição do objetivo
                    self.get_logger().info(f"Posição inicial: {self.start_position}, Objetivo: {self.goal_position}")
                else:
                    self.get_logger().error("Não foi possível encontrar as posições de início ou objetivo.")
                    self.start_position = None
                    self.goal_position = None
                
                # Converte o mapa para valores numéricos
                self.map_data = np.where(self.map_data == 'f', 0, 1)  # 0 = livre, 1 = obstáculo
                self.map_data[self.start_position] = 0  # Garante que o início seja acessível
                self.map_data[self.goal_position] = 0   # Garante que o objetivo seja acessível

            except Exception as e:
                self.get_logger().error(f"Erro ao processar mapa: {e}")
                self.map_data = None

            return self.map_data
        else:
            self.get_logger().error('Falha ao obter o mapa.')
            return None



    def a_star(self, start, goal):
        """Implementação do algoritmo A*."""
        def manhattan_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: manhattan_distance(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            if current == goal:
                # Reconstruindo o caminho
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            open_set.remove(current)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < self.map_data.shape[0] and
                    0 <= neighbor[1] < self.map_data.shape[1] and
                    self.map_data[neighbor] == 0):  # Célula livre
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + manhattan_distance(neighbor, goal)
                        open_set.add(neighbor)

        return []  # Nenhum caminho encontrado

    def plan_route(self):
        """Planeja a rota para o robô."""
        self.get_logger().info("Planejando a rota...")
        if self.map_data is not None and self.start_position and self.goal_position:
            self.path = self.a_star(self.start_position, self.goal_position)
            if self.path:
                self.get_logger().info(f"Rota planejada: {self.path}")
                # Exibe o caminho no mapa
                for x, y in self.path:
                    self.map_data[x, y] = 2  # Marca o caminho como 2
                self.get_logger().info(f"Mapa com rota:\n{self.map_data}")
            else:
                self.get_logger().error("Não foi possível planejar uma rota.")
        else:
            self.get_logger().error("Dados insuficientes para planejar a rota.")



    def navigate_path(self):
        """Navega pelo caminho planejado."""
        current_position = self.start_position  # Posição inicial do robô
        for target_position in self.path[1:]:  # Ignorar a posição inicial
            direction = self.get_direction_to(current_position, target_position)
            
            if not direction:
                self.get_logger().error(f"Não foi possível determinar a direção para {target_position}.")
                break  # Interromper em caso de erro

            # Mover o robô
            response = self.move_robot(direction)
            if response and not response.success:
                self.get_logger().error("Falha ao mover o robô.")
                break

            # Atualizar posição atual após o movimento
            current_position = target_position
            self.get_logger().info(f"Nova posição: {current_position}")
            
            # Adiciona um delay para visualizar a movimentação
            time.sleep(0.5)  # 0.5 segundos de delay

    def get_direction_to(self, current_position, target_position):
        """Determina a direção para o próximo passo."""
        dx = target_position[0] - current_position[0]  # Diferença em X
        dy = target_position[1] - current_position[1]  # Diferença em Y
        
        if dx == 0 and dy == 1:
            return 'right'
        elif dx == 0 and dy == -1:
            return 'left'
        elif dx == 1 and dy == 0:
            return 'down'
        elif dx == -1 and dy == 0:
            return 'up'
        else:
            # Caso a direção não seja válida
            self.get_logger().error(f"Direção inválida calculada: dx={dx}, dy={dy}")
            return None


    def move_robot(self, direction):
        """Move o robô em uma direção específica."""
        request = MoveCmd.Request()
        request.direction = direction
        future = self.move_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response:
            self.get_logger().info(f"Movimento: {direction} | Resultado: {response.success}")
            return response
        else:
            self.get_logger().error("Falha ao mover o robô.")
            return None

def main(args=None):
    rclpy.init(args=args)
    client = RobotClient()
    client.get_map()  # Obtém o mapa
    client.plan_route()  # Planeja a rota com A*
    client.navigate_path()  # Executa a rota planejada
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
