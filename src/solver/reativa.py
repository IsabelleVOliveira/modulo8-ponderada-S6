import time  # Importado para adicionar o delay
import rclpy
from rclpy.node import Node
from cg_interfaces.srv import MoveCmd

class ReactiveNavigation(Node):
    def __init__(self):
        super().__init__('reactive_navigation')
        self.move_client = self.create_client(MoveCmd, '/move_command')
        while not self.move_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Aguardando o serviço /move_command...')
        
        self.visited_positions = {}  # {posição: número de visitas}
        self.wall_side = 'left'  # Pode ser 'left' ou 'right'
        self.total_moves = 0  # Contador para o número total de movimentos

    def send_move_request(self, direction):
        """Envia uma solicitação de movimento na direção especificada."""
        request = MoveCmd.Request()
        request.direction = direction
        future = self.move_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def wall_following(self, sensors):
        """Implementa o algoritmo de seguir a parede."""
        if self.wall_side == 'left':
            if sensors['left'] == 'f':  # Parede à esquerda; seguir a parede
                return 'left'
            elif sensors['down'] == 'f':  # Frente livre; avançar
                return 'down'
            elif sensors['right'] == 'f':  # Vire à direita
                return 'right'
            elif sensors['up'] == 'f':  # Se tudo bloqueado, vá para cima
                return 'up'
        elif self.wall_side == 'right':
            if sensors['right'] == 'f':  # Parede à direita; seguir a parede
                return 'right'
            elif sensors['down'] == 'f':  # Frente livre; avançar
                return 'down'
            elif sensors['left'] == 'f':  # Vire à esquerda
                return 'left'
            elif sensors['up'] == 'f':  # Se tudo bloqueado, vá para cima
                return 'up'
        return None

    def get_next_direction(self, robot_pos, target_pos, sensors):
        """
        Combina menor frequência de visitas e Wall Following.
        """
        directions = {
            'left': (0, -1),
            'right': (0, 1),
            'up': (-1, 0),
            'down': (1, 0),
        }
        possible_moves = []

        for direction, (dx, dy) in directions.items():
            if sensors[direction] in ['f', 't']:  # Espaço livre ou alvo
                new_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
                visit_count = self.visited_positions.get(new_pos, 0)
                possible_moves.append((direction, new_pos, visit_count))

        # Priorizar células menos visitadas
        if possible_moves:
            possible_moves.sort(
                key=lambda x: (x[2], abs(x[1][0] - target_pos[0]) + abs(x[1][1] - target_pos[1]))
            )
            return possible_moves[0][0]

        # Caso nenhum movimento claro, usar Wall Following
        return self.wall_following(sensors)

    def reactive_navigation(self):
        """Implementa a lógica de navegação reativa aprimorada."""
        while rclpy.ok():
            result = self.send_move_request('')  # Solicitação para obter dados de sensores
            robot_pos = tuple(result.robot_pos)
            target_pos = tuple(result.target_pos)
            sensors = {
                'left': result.left,
                'down': result.down,
                'up': result.up,
                'right': result.right,
            }

            # Checa se chegou ao alvo
            if robot_pos == target_pos:
                self.get_logger().info('Alvo alcançado!')
                break

            # Atualiza o contador de visitas da posição atual
            self.visited_positions[robot_pos] = self.visited_positions.get(robot_pos, 0) + 1
            self.total_moves += 1  # Incrementa o total de movimentos realizados

            # Decide a próxima direção
            direction = self.get_next_direction(robot_pos, target_pos, sensors)
            if direction:
                move_result = self.send_move_request(direction)
                if move_result.success:
                    self.get_logger().info(f"Movimento bem-sucedido para {direction}. Nova posição: {move_result.robot_pos}")
                else:
                    self.get_logger().info(f"Falha ao mover para {direction}. Tentando outra direção.")
            else:
                self.get_logger().info('Sem direções válidas. Robô preso.')
                break

        # Log do total de movimentos realizados
        self.get_logger().info(f"Total de movimentos realizados (incluindo repetidos): {self.total_moves}")

        # Adiciona um leve delay para outros serviços ROS
        time.sleep(2)

def main():
    rclpy.init()
    navigator = ReactiveNavigation()
    navigator.reactive_navigation()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
