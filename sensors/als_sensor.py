import smbus2


class ALSSensor:

    I2C_BUS = 1
    DEVICE_ADDR = 0x32

    LUX_REG = 0x06
    CCT_REG = 0x07

    def __init__(self):
        self.bus = smbus2.SMBus(self.I2C_BUS)

    def read_register(self, register):
        data = self.bus.read_i2c_block_data(
            self.DEVICE_ADDR,
            register,
            2
        )
        return (data[0] << 8) | data[1]

    def read(self):
        lux = self.read_register(self.LUX_REG)
        cct = self.read_register(self.CCT_REG)

        return lux, cct

    def close(self):
        self.bus.close()