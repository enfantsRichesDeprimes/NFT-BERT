/**
 *Submitted for verification at Etherscan.io on 2022-02-08
*/

// SPDX-License-Identifier: MIT

// File: @openzeppelin/contracts/GSN/Context.sol

pragma solidity ^0.8.0;

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
  function _msgSender() internal view virtual returns (address) {
    return msg.sender;
  }

  function _msgData() internal view virtual returns (bytes calldata) {
    return msg.data;
  }
}

/**
 * @dev Interface of the ERC165 standard, as defined in the
 * https://eips.ethereum.org/EIPS/eip-165[EIP].
 *
 * Implementers can declare support of contract interfaces, which can then be
 * queried by others ({ERC165Checker}).
 *
 * For an implementation, see {ERC165}.
 */
interface IERC165 {
  /**
   * @dev Returns true if this contract implements the interface defined by
   * `interfaceId`. See the corresponding
   * https://eips.ethereum.org/EIPS/eip-165#how-interfaces-are-identified[EIP section]
   * to learn more about how these ids are created.
   *
   * This function call must use less than 30 000 gas.
   */
  function supportsInterface(bytes4 interfaceId) external view returns (bool);
}

/**
 * @title ERC721 token receiver interface
 * @dev Interface for any contract that wants to support safeTransfers
 * from ERC721 asset contracts.
 */
interface IERC721Receiver {
  /**
   * @dev Whenever an {IERC721} `tokenId` token is transferred to this contract via {IERC721-safeTransferFrom}
   * by `operator` from `from`, this function is called.
   *
   * It must return its Solidity selector to confirm the token transfer.
   * If any other value is returned or the interface is not implemented by the recipient, the transfer will be reverted.
   *
   * The selector can be obtained in Solidity with `IERC721.onERC721Received.selector`.
   */
  function onERC721Received(
    address operator,
    address from,
    uint256 tokenId,
    bytes calldata data
  ) external returns (bytes4);
}

// File: @openzeppelin/contracts/token/ERC721/IERC721.sol

/**
 * @dev Required interface of an ERC721 compliant contract.
 */
interface IERC721 is IERC165 {
  /**
   * @dev Emitted when `tokenId` token is transferred from `from` to `to`.
   */
  event Transfer(
    address indexed from,
    address indexed to,
    uint256 indexed tokenId
  );

  /**
   * @dev Emitted when `owner` enables `approved` to manage the `tokenId` token.
   */
  event Approval(
    address indexed owner,
    address indexed approved,
    uint256 indexed tokenId
  );

  /**
   * @dev Emitted when `owner` enables or disables (`approved`) `operator` to manage all of its assets.
   */
  event ApprovalForAll(
    address indexed owner,
    address indexed operator,
    bool approved
  );

  /**
   * @dev Returns the number of tokens in ``owner``'s account.
   */
  function balanceOf(address owner) external view returns (uint256 balance);

  /**
   * @dev Returns the owner of the `tokenId` token.
   *
   * Requirements:
   *
   * - `tokenId` must exist.
   */
  function ownerOf(uint256 tokenId) external view returns (address owner);

  /**
   * @dev Safely transfers `tokenId` token from `from` to `to`, checking first that contract recipients
   * are aware of the ERC721 protocol to prevent tokens from being forever locked.
   *
   * Requirements:
   *
   * - `from` cannot be the zero address.
   * - `to` cannot be the zero address.
   * - `tokenId` token must exist and be owned by `from`.
   * - If the caller is not `from`, it must be have been allowed to move this token by either {approve} or {setApprovalForAll}.
   * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
   *
   * Emits a {Transfer} event.
   */
  function safeTransferFrom(
    address from,
    address to,
    uint256 tokenId
  ) external;

  /**
   * @dev Safely transfers `tokenId` token from `from` to `to`.
   *
   * Requirements:
   *
   * - `from` cannot be the zero address.
   * - `to` cannot be the zero address.
   * - `tokenId` token must exist and be owned by `from`.
   * - If the caller is not `from`, it must be approved to move this token by either {approve} or {setApprovalForAll}.
   * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
   *
   * Emits a {Transfer} event.
   */
  function safeTransferFrom(
    address from,
    address to,
    uint256 tokenId,
    bytes calldata data
  ) external;

  /**
   * @dev Transfers `tokenId` token from `from` to `to`.
   *
   * WARNING: Usage of this method is discouraged, use {safeTransferFrom} whenever possible.
   *
   * Requirements:
   *
   * - `from` cannot be the zero address.
   * - `to` cannot be the zero address.
   * - `tokenId` token must be owned by `from`.
   * - If the caller is not `from`, it must be approved to move this token by either {approve} or {setApprovalForAll}.
   *
   * Emits a {Transfer} event.
   */
  function transferFrom(
    address from,
    address to,
    uint256 tokenId
  ) external;

  /**
   * @dev Gives permission to `to` to transfer `tokenId` token to another account.
   * The approval is cleared when the token is transferred.
   *
   * Only a single account can be approved at a time, so approving the zero address clears previous approvals.
   *
   * Requirements:
   *
   * - The caller must own the token or be an approved operator.
   * - `tokenId` must exist.
   *
   * Emits an {Approval} event.
   */
  function approve(address to, uint256 tokenId) external;

  /**
   * @dev Returns the account approved for `tokenId` token.
   *
   * Requirements:
   *
   * - `tokenId` must exist.
   */
  function getApproved(uint256 tokenId)
    external
    view
    returns (address operator);

  /**
   * @dev Approve or remove `operator` as an operator for the caller.
   * Operators can call {transferFrom} or {safeTransferFrom} for any token owned by the caller.
   *
   * Requirements:
   *
   * - The `operator` cannot be the caller.
   *
   * Emits an {ApprovalForAll} event.
   */
  function setApprovalForAll(address operator, bool _approved) external;

  /**
   * @dev Returns if the `operator` is allowed to manage all of the assets of `owner`.
   *
   * See {setApprovalForAll}
   */
  function isApprovedForAll(address owner, address operator)
    external
    view
    returns (bool);
}

// File: @openzeppelin/contracts/token/ERC721/IERC721Metadata.sol

/**
 * @title ERC-721 Non-Fungible Token Standard, optional metadata extension
 * @dev See https://eips.ethereum.org/EIPS/eip-721
 */
interface IERC721Metadata is IERC721 {
  /**
   * @dev Returns the token collection name.
   */
  function name() external view returns (string memory);

  /**
   * @dev Returns the token collection symbol.
   */
  function symbol() external view returns (string memory);

  /**
   * @dev Returns the Uniform Resource Identifier (URI) for `tokenId` token.
   */
  function tokenURI(uint256 tokenId) external view returns (string memory);
}

// File: @openzeppelin/contracts/introspection/ERC165.sol

/**
 * @dev Implementation of the {IERC165} interface.
 *
 * Contracts that want to implement ERC165 should inherit from this contract and override {supportsInterface} to check
 * for the additional interface id that will be supported. For example:
 *
 * ```solidity
 * function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
 *     return interfaceId == type(MyInterface).interfaceId || super.supportsInterface(interfaceId);
 * }
 * ```
 *
 * Alternatively, {ERC165Storage} provides an easier to use but more expensive implementation.
 */
abstract contract ERC165 is IERC165 {
  /**
   * @dev See {IERC165-supportsInterface}.
   */
  function supportsInterface(bytes4 interfaceId)
    public
    view
    virtual
    override
    returns (bool)
  {
    return interfaceId == type(IERC165).interfaceId;
  }
}

// File: @openzeppelin/contracts/utils/Strings.sol

/**
 * @dev String operations.
 */
library Strings {
  bytes16 private constant _HEX_SYMBOLS = '0123456789abcdef';

  /**
   * @dev Converts a `uint256` to its ASCII `string` decimal representation.
   */
  function toString(uint256 value) internal pure returns (string memory) {
    // Inspired by OraclizeAPI's implementation - MIT licence
    // https://github.com/oraclize/ethereum-api/blob/b42146b063c7d6ee1358846c198246239e9360e8/oraclizeAPI_0.4.25.sol

    if (value == 0) {
      return '0';
    }
    uint256 temp = value;
    uint256 digits;
    while (temp != 0) {
      digits++;
      temp /= 10;
    }
    bytes memory buffer = new bytes(digits);
    while (value != 0) {
      digits -= 1;
      buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
      value /= 10;
    }
    return string(buffer);
  }

  /**
   * @dev Converts a `uint256` to its ASCII `string` hexadecimal representation.
   */
  function toHexString(uint256 value) internal pure returns (string memory) {
    if (value == 0) {
      return '0x00';
    }
    uint256 temp = value;
    uint256 length = 0;
    while (temp != 0) {
      length++;
      temp >>= 8;
    }
    return toHexString(value, length);
  }

  /**
   * @dev Converts a `uint256` to its ASCII `string` hexadecimal representation with fixed length.
   */
  function toHexString(uint256 value, uint256 length)
    internal
    pure
    returns (string memory)
  {
    bytes memory buffer = new bytes(2 * length + 2);
    buffer[0] = '0';
    buffer[1] = 'x';
    for (uint256 i = 2 * length + 1; i > 1; --i) {
      buffer[i] = _HEX_SYMBOLS[value & 0xf];
      value >>= 4;
    }
    require(value == 0, 'Strings: ERROR');
    return string(buffer);
  }
}

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * By default, the owner account will be the one that deploys the contract. This
 * can later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
  address private _owner;

  string private constant ERR = 'Ownable: ERROR';

  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );

  /**
   * @dev Initializes the contract setting the deployer as the initial owner.
   */
  constructor() {
    _transferOwnership(_msgSender());
  }

  /**
   * @dev Returns the address of the current owner.
   */
  function owner() public view virtual returns (address) {
    return _owner;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(owner() == _msgSender(), ERR);
    _;
  }

  /**
   * @dev Leaves the contract without owner. It will not be possible to call
   * `onlyOwner` functions anymore. Can only be called by the current owner.
   *
   * NOTE: Renouncing ownership will leave the contract without an owner,
   * thereby removing any functionality that is only available to the owner.
   */
  function renounceOwnership() public virtual onlyOwner {
    _transferOwnership(address(0));
  }

  /**
   * @dev Transfers ownership of the contract to a new account (`newOwner`).
   * Can only be called by the current owner.
   */
  function transferOwnership(address newOwner) public virtual onlyOwner {
    require(newOwner != address(0), ERR);
    _transferOwnership(newOwner);
  }

  /**
   * @dev Transfers ownership of the contract to a new account (`newOwner`).
   * Internal function without access restriction.
   */
  function _transferOwnership(address newOwner) internal virtual {
    address oldOwner = _owner;
    _owner = newOwner;
    emit OwnershipTransferred(oldOwner, newOwner);
  }
}

library Address {
  /**
   * @dev Returns true if `account` is a contract.
   */
  function isContract(address account) internal view returns (bool) {
    // This method relies on extcodesize/address.code.length, which returns 0
    // for contracts in construction, since the code is only stored at the end
    // of the constructor execution.

    return account.code.length > 0;
  }
}

// File: @openzeppelin/contracts/token/ERC721/ERC721.sol

/**
 * @dev Implementation of https://eips.ethereum.org/EIPS/eip-721[ERC721] Non-Fungible Token Standard, including
 * the Metadata extension, but not including the Enumerable extension, which is available separately as
 * {ERC721Enumerable}.
 */
contract ERC721 is Context, Ownable, ERC165, IERC721, IERC721Metadata {
  using Strings for uint256;
  using Address for address;

  string private constant ERR = 'ERC721: ERROR';

  // Token name
  string internal _name;

  // Token symbol
  string internal _symbol;

  // Base URI
  string private _baseURI;

  // Current owner address index
  uint256 _ownerCounter = 0;

  // Mapping from token ID to 16 owner addresses
  mapping(uint256 => uint256) private _owners;

  // Mapping owner address to internal owner id + balances
  mapping(address => uint256) private _extToIntMap;

  // Mapping internal address to external address
  mapping(uint256 => address) private _intToExtMap;

  // Mapping from token ID to approved address
  mapping(uint256 => address) private _tokenApprovals;

  // Mapping from owner to operator approvals
  mapping(address => mapping(address => bool)) private _operatorApprovals;

  /**
   * @dev Initializes the contract by setting a `name` and a `symbol` to the token collection.
   */
  function _initialize(string memory name_, string memory symbol_) internal {
    _name = name_;
    _symbol = symbol_;
  }

  /**
   * @dev See {IERC165-supportsInterface}.
   */
  function supportsInterface(bytes4 interfaceId)
    public
    view
    virtual
    override(ERC165, IERC165)
    returns (bool)
  {
    return
      interfaceId == type(IERC721).interfaceId ||
      interfaceId == type(IERC721Metadata).interfaceId ||
      super.supportsInterface(interfaceId);
  }

  /**
   * @dev See {IERC721-balanceOf}.
   */
  function balanceOf(address owner)
    public
    view
    virtual
    override
    returns (uint256)
  {
    require(owner != address(0), ERR);
    return _extToIntMap[owner] >> 128;
  }

  /**
   * @dev See {IERC721-ownerOf}.
   */
  function ownerOf(uint256 tokenId)
    public
    view
    virtual
    override
    returns (address)
  {
    uint256 internalOwner = _getInternalOwner(tokenId);
    require(internalOwner != 0, ERR);
    return _intToExtMap[internalOwner];
  }

  /**
   * @dev See {IERC721Metadata-name}.
   */
  function name() public view virtual override returns (string memory) {
    return _name;
  }

  /**
   * @dev See {IERC721Metadata-symbol}.
   */
  function symbol() public view virtual override returns (string memory) {
    return _symbol;
  }

  /**
   * @dev See {IERC721Metadata-tokenURI}.
   */
  function tokenURI(uint256 tokenId)
    public
    view
    virtual
    override
    returns (string memory)
  {
    require(_getInternalOwner(tokenId) != 0, ERR);

    bytes memory bytesURI = bytes(_baseURI);
    if (bytesURI.length == 0 || bytesURI[bytesURI.length - 1] == '/')
      return string(abi.encodePacked(_baseURI, tokenId.toString(), '.json'));
    else return _baseURI;
  }

  function setBaseURI(string memory newBaseURI) external onlyOwner {
    _baseURI = newBaseURI;
  }

  /**
   * @dev See {IERC721-approve}.
   */
  function approve(address to, uint256 tokenId) public virtual override {
    address owner = ERC721.ownerOf(tokenId);
    require(to != owner, ERR);

    require(
      _msgSender() == owner || isApprovedForAll(owner, _msgSender()),
      ERR
    );
    _approve(to, tokenId);
  }

  /**
   * @dev See {IERC721-getApproved}.
   */
  function getApproved(uint256 tokenId)
    public
    view
    virtual
    override
    returns (address)
  {
    require(_getInternalOwner(tokenId) != 0, ERR);

    return _tokenApprovals[tokenId];
  }

  /**
   * @dev See {IERC721-setApprovalForAll}.
   */
  function setApprovalForAll(address operator, bool approved)
    public
    virtual
    override
  {
    _setApprovalForAll(_msgSender(), operator, approved);
  }

  /**
   * @dev See {IERC721-isApprovedForAll}.
   */
  function isApprovedForAll(address owner, address operator)
    public
    view
    virtual
    override
    returns (bool)
  {
    return _operatorApprovals[owner][operator];
  }

  /**
   * @dev See {IERC721-transferFrom}.
   */
  function transferFrom(
    address from,
    address to,
    uint256 tokenId
  ) public virtual override {
    //solhint-disable-next-line max-line-length
    require(_isApprovedOrOwner(_msgSender(), tokenId), ERR);

    _transfer(from, to, tokenId);
  }

  /**
   * @dev See {IERC721-safeTransferFrom}.
   */
  function safeTransferFrom(
    address from,
    address to,
    uint256 tokenId
  ) public virtual override {
    safeTransferFrom(from, to, tokenId, '');
  }

  /**
   * @dev See {IERC721-safeTransferFrom}.
   */
  function safeTransferFrom(
    address from,
    address to,
    uint256 tokenId,
    bytes memory _data
  ) public virtual override {
    require(_isApprovedOrOwner(_msgSender(), tokenId), ERR);
    _safeTransfer(from, to, tokenId, _data);
  }

  /**
   * @dev Safely transfers `tokenId` token from `from` to `to`, checking first that contract recipients
   * are aware of the ERC721 protocol to prevent tokens from being forever locked.
   *
   * `_data` is additional data, it has no specified format and it is sent in call to `to`.
   *
   * This internal function is equivalent to {safeTransferFrom}, and can be used to e.g.
   * implement alternative mechanisms to perform token transfer, such as signature-based.
   *
   * Requirements:
   *
   * - `from` cannot be the zero address.
   * - `to` cannot be the zero address.
   * - `tokenId` token must exist and be owned by `from`.
   * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
   *
   * Emits a {Transfer} event.
   */
  function _safeTransfer(
    address from,
    address to,
    uint256 tokenId,
    bytes memory _data
  ) internal virtual {
    _transfer(from, to, tokenId);
    require(_checkOnERC721Received(from, to, tokenId, _data), ERR);
  }

  /**
   * @dev Returns whether `spender` is allowed to manage `tokenId`.
   *
   * Requirements:
   *
   * - `tokenId` must exist.
   */
  function _isApprovedOrOwner(address spender, uint256 tokenId)
    internal
    view
    virtual
    returns (bool)
  {
    require(_getInternalOwner(tokenId) != 0, ERR);
    address owner = ERC721.ownerOf(tokenId);
    return (spender == owner ||
      getApproved(tokenId) == spender ||
      isApprovedForAll(owner, spender));
  }

  /**
   * @dev Mints `tokenId` and transfers it to `to`.
   *
   * WARNING: Usage of this method is discouraged, use {_safeMint} whenever possible
   *
   * Requirements:
   *
   * - `tokenId` must not exist.
   * - `to` cannot be the zero address.
   *
   * Emits a {Transfer} event.
   */
  function _mint(
    address to,
    uint256 tokenId,
    uint256 num
  ) internal virtual {
    require(to != address(0), ERR);

    uint256 toInt = _getInternalOwnerFromAddress(to);
    uint256 tokenIdEnd = tokenId + num;
    uint256 curBase = tokenId >> 4;
    uint256 mask = _owners[curBase];

    for (; tokenId < tokenIdEnd; ++tokenId) {
      // Update storage balance of previous bin
      uint256 base = tokenId >> 4;
      uint256 idBits = (tokenId & 0xF) << 4;
      if (base != curBase) {
        _owners[curBase] = mask;
        curBase = base;
        mask = _owners[curBase];
      }
      require(((mask >> idBits) & 0xFFFF) == 0, ERR);
      mask |= (toInt << idBits);

      emit Transfer(address(0), to, tokenId);
    }
    _owners[curBase] = mask;
    _extToIntMap[to] += num << 128;
  }

  /**
   * @dev Destroys `tokenId`.
   * The approval is cleared when the token is burned.
   *
   * Requirements:
   *
   * - `tokenId` must exist.
   *
   * Emits a {Transfer} event.
   */
  function _burn(uint256 tokenId) internal virtual {
    // Clear approvals
    _approve(address(0), tokenId);

    uint256 intOwner = _getInternalOwner(tokenId);
    require(intOwner != 0, ERR);
    _setInternalOwner(tokenId, 0);

    address owner = _intToExtMap[intOwner];
    _extToIntMap[owner] -= 1 << 128;

    emit Transfer(owner, address(0), tokenId);
  }

  /**
   * @dev Transfers `tokenId` from `from` to `to`.
   *  As opposed to {transferFrom}, this imposes no restrictions on msg.sender.
   *
   * Requirements:
   *
   * - `to` cannot be the zero address.
   * - `tokenId` token must be owned by `from`.
   *
   * Emits a {Transfer} event.
   */
  function _transfer(
    address from,
    address to,
    uint256 tokenId
  ) internal virtual {
    uint256 intOwner = _getInternalOwner(tokenId);
    require(_intToExtMap[intOwner] == from, ERR);
    require(to != address(0), ERR);

    // Clear approvals from the previous owner
    _approve(address(0), tokenId);

    uint256 toInt = _getInternalOwnerFromAddress(to);
    _setInternalOwner(tokenId, toInt);

    _extToIntMap[from] -= 1 << 128;
    _extToIntMap[to] += 1 << 128;

    emit Transfer(from, to, tokenId);
  }

  /**
   * @dev Internal function to invoke {IERC721Receiver-onERC721Received} on a target address.
   * The call is not executed if the target address is not a contract.
   *
   * @param from address representing the previous owner of the given token ID
   * @param to target address that will receive the tokens
   * @param tokenId uint256 ID of the token to be transferred
   * @param _data bytes optional data to send along with the call
   * @return bool whether the call correctly returned the expected magic value
   */
  function _checkOnERC721Received(
    address from,
    address to,
    uint256 tokenId,
    bytes memory _data
  ) private returns (bool) {
    if (to.isContract()) {
      try
        IERC721Receiver(to).onERC721Received(_msgSender(), from, tokenId, _data)
      returns (bytes4 retval) {
        return retval == IERC721Receiver.onERC721Received.selector;
      } catch (bytes memory reason) {
        if (reason.length == 0) {
          revert(ERR);
        } else {
          assembly {
            revert(add(32, reason), mload(reason))
          }
        }
      }
    } else {
      return true;
    }
  }

  /**
   * @dev Approve `to` to operate on `tokenId`
   *
   * Emits a {Approval} event.
   */
  function _approve(address to, uint256 tokenId) internal virtual {
    _tokenApprovals[tokenId] = to;
    emit Approval(ERC721.ownerOf(tokenId), to, tokenId);
  }

  /**
   * @dev Approve `operator` to operate on all of `owner` tokens
   *
   * Emits a {ApprovalForAll} event.
   */
  function _setApprovalForAll(
    address owner,
    address operator,
    bool approved
  ) internal virtual {
    require(owner != operator, ERR);
    _operatorApprovals[owner][operator] = approved;
    emit ApprovalForAll(owner, operator, approved);
  }

  function _getInternalOwner(uint256 tokenId) internal view returns (uint256) {
    return (_owners[tokenId >> 4] >> ((tokenId & 0xF) << 4)) & 0xFFFF;
  }

  function _getInternalOwnerFromAddress(address externalOwner)
    internal
    returns (uint256)
  {
    uint256 intOwner = _extToIntMap[externalOwner];
    if (intOwner == 0) {
      require(_ownerCounter < 0xFFFF, ERR);
      _extToIntMap[externalOwner] = intOwner = ++_ownerCounter;
      _intToExtMap[intOwner] = externalOwner;
    }
    return uint256(uint128(intOwner));
  }

  function _setInternalOwner(uint256 tokenId, uint256 newOwner) internal {
    uint256 mask = _owners[tokenId >> 4] & ~(0xFFFF << ((tokenId & 0xF) << 4));
    _owners[tokenId >> 4] = mask | (newOwner << ((tokenId & 0xF) << 4));
  }
}

/**
 * @dev OpenSea proxy registry to prevent gas spend for approvals
 */
contract ProxyRegistry {
  mapping(address => address) public proxies;
}

/**
 * @dev Contract module that helps prevent reentrant calls to a function.
 *
 * Inheriting from `ReentrancyGuard` will make the {nonReentrant} modifier
 * available, which can be applied to functions to make sure there are no nested
 * (reentrant) calls to them.
 *
 * Note that because there is a single `nonReentrant` guard, functions marked as
 * `nonReentrant` may not call one another. This can be worked around by making
 * those functions `private`, and then adding `external` `nonReentrant` entry
 * points to them.
 *
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/[Reentrancy After Istanbul].
 */
abstract contract ReentrancyGuard {
  // Booleans are more expensive than uint256 or any type that takes up a full
  // word because each write operation emits an extra SLOAD to first read the
  // slot's contents, replace the bits taken up by the boolean, and then write
  // back. This is the compiler's defense against contract upgrades and
  // pointer aliasing, and it cannot be disabled.

  // The values being non-zero value makes deployment a bit more expensive,
  // but in exchange the refund on every call to nonReentrant will be lower in
  // amount. Since refunds are capped to a percentage of the total
  // transaction's gas, it is best to keep them low in cases like this one, to
  // increase the likelihood of the full refund coming into effect.
  uint256 private constant _NOT_ENTERED = 1;
  uint256 private constant _ENTERED = 2;

  uint256 private _status;

  constructor() {
    _status = _NOT_ENTERED;
  }

  /**
   * @dev Prevents a contract from calling itself, directly or indirectly.
   * Calling a `nonReentrant` function from another `nonReentrant`
   * function is not supported. It is possible to prevent this from happening
   * by making the `nonReentrant` function external, and making it call a
   * `private` function that does the actual work.
   */
  modifier nonReentrant() {
    // On the first call to nonReentrant, _notEntered will be true
    require(_status != _ENTERED, 'ReentrancyGuard: ERROR');

    // Any calls to nonReentrant after this point will fail
    _status = _ENTERED;

    _;

    // By storing the original value once again, a refund is triggered (see
    // https://eips.ethereum.org/EIPS/eip-2200)
    _status = _NOT_ENTERED;
  }
}

/**
 * @dev Implementation of Opt ERC721 contract
 */
contract ERC721Opt is ERC721, ReentrancyGuard {
  string private constant ERR = 'ERC721Opt: Error';

  // OpenSea proxy registry
  address private immutable _osProxyRegistryAddress;
  // Address allowed to initialize contract
  address private immutable _initializer;

  // Max mints per transaction
  uint256 private _maxTxMint;

  // The CAP of mintable tokenIds
  uint256 private _cap;

  // The CAP of free mintable tokenIds
  uint256 private _freeCap;

  // ETH price of one tokenIds
  uint256 private _tokenPrice;

  // TokenId counter, 1 minted in ctor
  uint256 private _currentTokenId;

  // Mint Running
  uint256 public mintRunning;

  // Whitelist minters
  mapping(address => uint256) private _wlMinted;

  // Gift minting
  address[] private giftId2Address;

  // Fired when funds are distributed
  event Withdraw(address indexed receiver, uint256 amount);

  // Whitelist address check
  modifier onlyValidAddress(
    uint256 _salt,
    bytes32 _r,
    bytes32 _s,
    uint8 _v
  ) {
    bytes32 hash = keccak256(abi.encode(address(this), _salt, _msgSender()));
    bytes32 message = keccak256(
      abi.encodePacked('\x19Ethereum Signed Message:\n32', hash)
    );
    address sig = ecrecover(message, _v, _r, _s);

    require(owner() == sig, ERR);
    _;
  }

  /**
   * @dev Initialization.
   */
  constructor(address initializer_, address osProxyRegistry_) {
    _osProxyRegistryAddress = osProxyRegistry_;
    _initializer = initializer_;
  }

  /**
   * @dev Clone Initialization.
   */
  function initialize(
    address owner_,
    string memory name_,
    string memory symbol_,
    uint256 cap_,
    uint256 freeCap_,
    uint256 maxPerTx_,
    uint256 price_
  ) external {
    require(_msgSender() == _initializer, ERR);

    _transferOwnership(owner_);

    ERC721._initialize(name_, symbol_);
    _cap = cap_;
    _freeCap = freeCap_ + 1;
    _maxTxMint = maxPerTx_;
    _tokenPrice = price_;
    _currentTokenId = 1;

    emit Transfer(address(0), address(0), 0);
  }

  /**
   * @dev See {IERC721-isApprovedForAll}.
   */
  function isApprovedForAll(address owner, address operator)
    public
    view
    virtual
    override
    returns (bool)
  {
    // Whitelist OpenSea proxy contract for easy trading.
    ProxyRegistry proxyRegistry = ProxyRegistry(_osProxyRegistryAddress);
    if (address(proxyRegistry.proxies(owner)) == operator) {
      return true;
    }
    return super.isApprovedForAll(owner, operator);
  }

  /**
   * @dev See {ERC721-tokenURI}.
   */
  function tokenURI(uint256 tokenId)
    public
    view
    virtual
    override
    returns (string memory)
  {
    return super.tokenURI(tokenId < 100000 ? tokenId : 0);
  }

  /**
   * @dev See {ERC721-ownerOf}.
   */
  function ownerOf(uint256 tokenId)
    public
    view
    virtual
    override
    returns (address)
  {
    if (tokenId < 100000) return super.ownerOf(tokenId);
    return (giftId2Address[(tokenId - 100000) / 1000]);
  }

  /**
   * @dev mint
   */
  function mint(address to, uint256 numMint) external payable nonReentrant {
    require(mintRunning > 0, ERR);

    uint256 tidEnd = _currentTokenId + numMint;

    uint256 numPayMint = tidEnd > _freeCap ? tidEnd - _freeCap : 0;
    if (numPayMint > numMint) numPayMint = numMint;

    require(
      numMint > 0 &&
        numMint <= _maxTxMint &&
        tidEnd <= _cap &&
        msg.value >= numPayMint * _tokenPrice,
      ERR
    );

    _mint(to, _currentTokenId, numMint);
    _currentTokenId += numMint;

    {
      uint256 dust = msg.value - (numPayMint * _tokenPrice);
      if (dust > 0) payable(_msgSender()).transfer(dust);
    }
  }

  /**
   * @dev Whitelist mint a set of tokenIds for caller
   */
  function whitelistMint(
    uint256 numMint,
    uint256 compensation,
    bytes32 r,
    bytes32 s,
    uint8 v
  )
    external
    nonReentrant
    onlyValidAddress(numMint | (compensation << 128), r, s, v)
  {
    // Only once per wallet
    require(_wlMinted[_msgSender()] == 0, ERR);

    _wlMinted[_msgSender()] = 1;
    _mint(_msgSender(), _currentTokenId, numMint);
    _currentTokenId += numMint;

    if (compensation > 0 && address(this).balance >= compensation) {
      require((compensation >> 128) == 0, ERR);
      payable(_msgSender()).transfer(compensation);
    }
  }

  /**
   * @dev mint gift
   */
  function mint(
    address[] calldata to,
    uint256[] calldata startMints,
    uint256[] calldata numMints
  ) external onlyOwner {
    for (uint256 i = 0; i < to.length; ++i) {
      uint256 nextMint = startMints[i];
      if (nextMint == 0) {
        nextMint = 100000 + giftId2Address.length * 1000;
        giftId2Address.push(to[i]);
      }
      uint256 endMint = nextMint + numMints[i];
      for (; nextMint < endMint; ++nextMint) {
        emit Transfer(address(0), to[i], nextMint);
      }
    }
  }

  /**
   * @dev Withdraw rewards
   */
  function withdraw(address account) external onlyOwner {
    uint256 amount = address(this).balance;
    payable(account).transfer(amount);
    emit Withdraw(account, amount);
  }

  /**
   * @dev return max number token mintable
   */
  function totalSupply() external view returns (uint256) {
    return _cap;
  }

  /**
   * @dev return number of minted token
   */
  function tokenMinted() external view returns (uint256) {
    return _currentTokenId - 1;
  }

  /**
   * @dev return number of remaining free token
   */
  function freeMintLeft() external view returns (uint256) {
    return _currentTokenId < _freeCap ? _freeCap - _currentTokenId : 0;
  }

  /**
   * @dev return number of gift mint addresses
   */
  function giftAddressCount() external view returns (uint256) {
    return giftId2Address.length;
  }

  /**
   * @dev Set free mintable token cap
   */
  function setFreeTokenCap(uint256 newCap) external onlyOwner {
    _freeCap = newCap + 1;
  }

  /**
   * @dev Set token price
   */
  function setTokenPrice(uint256 newPrice) external onlyOwner {
    _tokenPrice = newPrice;
  }

  /**
   * @dev See Start / Stop minting.
   */
  function setMintRunning(uint256 newMintRunning) external onlyOwner {
    mintRunning = newMintRunning;
  }

  /**
   * @dev we don't allow ether receive()
   */
  receive() external payable {
    revert(ERR);
  }
}