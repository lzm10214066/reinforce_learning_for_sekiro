from environment.reward.memory import Memory


class RoleStatus:
    def __init__(self, base):
        self.current_health_address = base + 0x130
        self.max_health_address = self.current_health_address + 0x4
        self.current_HP_address = base + 0x148
        self.max_HP_address = self.current_HP_address + 0x4
        self.current_health = 0
        self.max_health = 0
        self.current_HP = 0
        self.max_HP = 0
        self.pre_health = 0
        self.pre_HP = 0


class RewardReader:
    def __init__(self, base_player, base_boss):
        self.m = Memory()
        process = "sekiro.exe"
        self.cHandle = self.m.GetProcessHandle(process, 0)

        self.player_status = RoleStatus(base_player)
        self.boss_status = RoleStatus(base_boss)
        self.read_current_status(self.player_status)
        self.read_current_status(self.boss_status)
        self.read_max_status(self.player_status)
        self.player_status.pre_health = self.player_status.current_health
        self.player_status.pre_HP = self.player_status.current_HP
        self.read_max_status(self.boss_status)
        self.boss_status.pre_health = self.boss_status.current_health
        self.boss_status.pre_HP = self.boss_status.current_HP

    def read_current_status(self, status):
        status.current_health = self.m.Read_UINT32(self.cHandle, status.current_health_address)
        status.current_HP = self.m.Read_UINT32(self.cHandle, status.current_HP_address)

    def change_value_to(self, address, v):
        self.m.Write_UINT32(self.cHandle, address, v)

    def read_max_status(self, status):
        status.max_health = self.m.Read_UINT32(self.cHandle, status.max_health_address)
        status.max_HP = self.m.Read_UINT32(self.cHandle, status.max_HP_address)

    def norm_v(self, current_v, max_v):
        return current_v / max_v

    def get_reward(self, action=None):
        self.read_current_status(self.player_status)
        self.read_current_status(self.boss_status)

        reward_player = self.norm_v(self.player_status.current_health - self.player_status.max_health,
                                    self.player_status.max_health) + 0 * self.norm_v(
            self.player_status.current_HP - self.player_status.max_HP, self.player_status.max_HP)
        self.change_value_to(self.player_status.current_health_address, self.player_status.max_health)
        self.change_value_to(self.player_status.current_HP_address, self.player_status.max_HP)

        r_hp_w = 1
        # if action == 0 or action == 1:
        #     r_hp_w = 1
        reward_boss = self.norm_v(self.boss_status.pre_health - self.boss_status.current_health,
                                  self.boss_status.max_health) + r_hp_w * self.norm_v(
            self.boss_status.pre_HP - self.boss_status.current_HP, self.boss_status.max_HP)
        self.boss_status.pre_health = self.boss_status.current_health
        self.boss_status.pre_HP = self.boss_status.current_HP

        boss_current_health_p = self.norm_v(self.boss_status.current_health, self.boss_status.max_health)
        boss_current_HP_p = self.norm_v(self.boss_status.current_HP, self.boss_status.max_HP)
        if boss_current_health_p < 0.1:
            self.change_value_to(self.boss_status.current_health_address, self.boss_status.max_health)
            self.boss_status.pre_health = self.boss_status.max_health

        if boss_current_HP_p < 0.1:
            self.change_value_to(self.boss_status.current_HP_address, self.boss_status.max_HP)
            self.boss_status.pre_HP = self.boss_status.max_HP

        r = reward_player + 10 * reward_boss
        if action != 4:
            r -= 0.0001
        return r


if __name__ == '__main__':
    base_player = 0x7ff4f74423f0
    base_boss = 0x7ff49eb297e0
    reward_reader = RewardReader(base_player, base_boss)
    while True:
        reward = reward_reader.get_reward(action=4)
        if abs(reward) > 0.000001:
            print('reward:', reward)
