import os
import shutil

herbivores_omnivores = {
    # Động vật có vú ăn cỏ/tạp
    'antelope', 'bison', 'cow', 'deer', 'donkey', 'elephant', 'goat', 'gorilla', 
    'hamster', 'hare', 'hippopotamus', 'horse', 'kangaroo', 'koala', 'ox', 'panda', 
    'pig', 'porcupine', 'reindeer', 'rhinoceros', 'sheep', 'squirrel', 'zebra',
    'bear', 'boar', 'chimpanzee', 'mouse', 'orangutan', 'raccoon', 'rat', 'wombat',
    'hedgehog', 'possum',
    
    # Chim ăn tạp/hạt/mật hoa
    'crow', 'duck', 'parrot', 'pigeon', 'turkey', 'flamingo', 'goose', 'hornbill', 
    'hummingbird', 'sparrow', 'swan', 'woodpecker', 'sandpiper', 'pelecaniformes',
    
    # Côn trùng ăn thực vật
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'dragonfly', 
    'fly', 'grasshopper', 'ladybugs', 'moth',

    # Động vật biển ăn tạp
    'lobster',    # Tôm hùm ăn cả thực vật và động vật nhỏ
    'goldfish',   # Cá vàng ăn tạp
    'turtle'      # Hầu hết các loài rùa ăn tạp
}

carnivores = {
    # Động vật có vú ăn thịt
    'badger', 'bat', 'cat', 'coyote', 'dog', 'dolphin', 'fox', 'hyena',
    'leopard', 'lion', 'otter', 'tiger', 'whale', 'wolf',
    
    # Chim săn mồi
    'eagle', 'owl', 'penguin',
    
    # Động vật biển ăn thịt
    'crab', 'jellyfish', 'octopus', 'seal', 'shark', 'squid', 'starfish',
    
    # Bò sát ăn thịt
    'lizard', 'snake',

    # Côn trùng/động vật biển ăn thịt
    'mosquito',     # Hút máu
    'oyster',       # Ăn sinh vật phù du
    'seahorse'      # Ăn động vật phù du nhỏ
}

# Cập nhật thêm okapi vào nhóm ăn cỏ
herbivores_omnivores.add('okapi')  # Họ hàng của hươu cao cổ, ăn lá cây

def organize_animals(base_path):
    # Tạo thư mục cho từng nhóm
    herbivores_path = os.path.join(base_path, 'herbivores_omnivores')
    carnivores_path = os.path.join(base_path, 'carnivores')
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(herbivores_path, exist_ok=True)
    os.makedirs(carnivores_path, exist_ok=True)
    
    # Duyệt qua các thư mục động vật
    for animal in os.listdir(base_path):
        animal_path = os.path.join(base_path, animal)
        
        # Bỏ qua nếu không phải thư mục hoặc là thư mục phân loại
        if not os.path.isdir(animal_path) or animal in ['herbivores_omnivores', 'carnivores']:
            continue
            
        # Di chuyển động vật vào thư mục tương ứng
        if animal in herbivores_omnivores:
            destination = os.path.join(herbivores_path, animal)
            shutil.move(animal_path, destination)
            print(f"Moved {animal} to herbivores_omnivores")
        elif animal in carnivores:
            destination = os.path.join(carnivores_path, animal)
            shutil.move(animal_path, destination)
            print(f"Moved {animal} to carnivores")
        else:
            print(f"Skipped {animal} - classification unknown")

if __name__ == "__main__":
    dataset_path = "dataset"
    organize_animals(dataset_path)